"""Verdicts over the CPU-offload decision journal.

WHY A SEPARATE TOOL FROM THE REPORT
-----------------------------------
``trace.py``'s exit report answers "what did this process do". That question
has no wrong answer, which is the problem: the defect this tooling was built
for was invisible inside a single run. Two runs of one prompt each made a
self-consistent set of choices and differed only from each other, so anything
that looks at one run in isolation reports a clean bill of health twice.

This module compares. It reads the journal ``trace.py`` streams -- one JSON
object per decision, keyed on the forward-pass counter -- and returns a
verdict rather than a rendering. The distinction matters: the previous
workflow was "print digests, let a human diff two reports", and a human
diffing two 500-line reports is exactly the step where "94 of 94 layers are
fine" got believed.

WHAT IT CAN DECIDE
------------------
Within one run, three properties must hold after the cadence and ownership
fixes, and each fails loudly rather than subtly:

  * re-selection fires on a fixed forward-pass stride. A varying stride means
    something other than the counter is driving the gate.
  * no two ranks may be assigned the same expert on the same layer in one
    event. That is double-compute by construction: masking removes only the
    offloading rank's own copy, so the second rank's GPU copy survives and
    the TP all-reduce sums both.
  * the event sequence is ordered -- ticks increment by one, the forward
    counter never goes backwards.

Between two runs, the first divergence is classified, because the three
classes need opposite responses:

  * SCHEDULER  -- same forward number, same traffic going in, different
                  choice coming out. The selector itself is nondeterministic.
  * UPSTREAM   -- same forward number, different traffic going in. The
                  scheduler is faithfully reflecting a divergence that
                  already happened in batching or routing; looking at the
                  selector will waste the afternoon.
  * GATE       -- same traffic and same choice, but at a different forward
                  number. The trigger moved. This is the shape of the
                  wall-clock bug: both runs decided the same thing, on
                  different steps, and the outputs still diverged because
                  the swap landed on a different token.

USAGE
-----
    SGLANG_MOE_TRACE=1 SGLANG_MOE_TRACE_JOURNAL=/tmp/a.jsonl  <run>
    SGLANG_MOE_TRACE=1 SGLANG_MOE_TRACE_JOURNAL=/tmp/b.jsonl  <run>

    python -m sglang.srt.eplb.cpu_offload.journal check /tmp/a.jsonl
    python -m sglang.srt.eplb.cpu_offload.journal diff  /tmp/a.jsonl /tmp/b.jsonl

Exit status is 0 when every check passes and the journals agree, 1 otherwise,
so this is usable as a build step rather than something someone remembers to
eyeball.
"""

from __future__ import annotations

import json
from typing import Any, Optional

SCHEMA_SUPPORTED = (1,)


class JournalError(Exception):
    """The file is not a journal this version knows how to read."""


def load(path: str) -> tuple[list[dict], int]:
    """Return ``(events, schema_version)``.

    A truncated final line is tolerated and dropped: the sink flushes per
    event, so a killed run leaves a valid prefix plus at most one partial
    line, and that prefix is the evidence worth having.
    """
    events: list[dict] = []
    schema: Optional[int] = None
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # partial trailing write from a killed run
            raise JournalError("%s: malformed record on line %d" % (path, i + 1))
        if rec.get("event") == "_schema":
            schema = int(rec.get("v", 0))
            continue
        events.append(rec)
    if schema is None:
        raise JournalError("%s: no schema record; not a decision journal" % path)
    if schema not in SCHEMA_SUPPORTED:
        raise JournalError(
            "%s: schema v%d, this tool understands %s. Refusing to compare "
            "rather than silently matching different fields."
            % (path, schema, list(SCHEMA_SUPPORTED))
        )
    return events, schema


# ---------------------------------------------------------------------------
# Single-run checks
# ---------------------------------------------------------------------------


def _check_ordering(events: list[dict]) -> list[str]:
    problems = []
    prev_fwd = -1
    prev_tick = None
    for ev in events:
        fwd = int(ev.get("forward_no", -1))
        if fwd < prev_fwd:
            problems.append(
                "forward counter went backwards: %d after %d (tick=%s). The "
                "counter is the only alignment axis; if it is not monotonic "
                "nothing downstream can be compared." % (fwd, prev_fwd, ev.get("tick"))
            )
        prev_fwd = fwd
        if ev.get("event") == "reselect":
            tick = int(ev.get("tick", -1))
            if prev_tick is not None and tick != prev_tick + 1:
                problems.append(
                    "re-selection tick jumped %d -> %d. Events are dropped or "
                    "duplicated, so a diff against another run would "
                    "mis-align." % (prev_tick, tick)
                )
            prev_tick = tick
    return problems


def _check_cadence(events: list[dict]) -> tuple[list[str], Optional[int]]:
    """Re-selection must fire on a fixed forward-pass stride."""
    fwds = [int(e["forward_no"]) for e in events if e.get("event") == "reselect"]
    if len(fwds) < 3:
        return [], None
    strides = {b - a for a, b in zip(fwds, fwds[1:])}
    if len(strides) == 1:
        return [], strides.pop()
    return (
        [
            "re-selection stride is not constant: saw %s. After the cadence fix "
            "the gate is a forward-pass counter, which yields one stride; several "
            "means something else is driving it (a clock, or a data-dependent "
            "early return upstream of the gate)." % sorted(strides)
        ],
        None,
    )


def _check_no_shared_expert(events: list[dict]) -> list[str]:
    """No two ranks may take the same expert on the same layer.

    Masking removes only the offloading rank's own GPU copy. If a second rank
    holds that expert it still computes it, and the TP all-reduce adds the CPU
    result on top -- the pair is counted twice. This is checkable from the
    journal alone, with no model metadata, because the event carries every
    rank's assignment.
    """
    problems = []
    for ev in events:
        xrank = ev.get("xrank")
        if not xrank:
            continue
        seen: dict[tuple, list] = {}
        for rank_id, changes in xrank.items():
            for layer_id, _old, new in changes:
                if new < 0:
                    continue
                seen.setdefault((layer_id, new), []).append(int(rank_id))
        for (layer_id, expert_id), ranks in sorted(seen.items()):
            if len(ranks) > 1:
                problems.append(
                    "fwd=%s layer=%d expert=%d assigned to ranks %s at once. "
                    "Every token routed there is computed twice."
                    % (ev.get("forward_no"), layer_id, expert_id, sorted(ranks))
                )
    return problems


def check(path: str) -> tuple[bool, list[str]]:
    """Run every single-journal check. Returns ``(ok, lines)``."""
    events, schema = load(path)
    out = ["journal: %s" % path, "schema: v%d" % schema]
    resel = [e for e in events if e.get("event") == "reselect"]
    boot = [e for e in events if e.get("event") == "bootstrap"]
    out.append(
        "events: %d (%d bootstrap, %d re-selection)"
        % (len(events), len(boot), len(resel))
    )
    if not events:
        out.append("")
        out.append(
            "EMPTY -- no decision was ever recorded. Either the fast "
            "loop never ran, or re-selection is pinned off. That is a "
            "valid configuration, but it is not evidence of "
            "determinism: nothing had the chance to vary."
        )
        return True, out

    problems: list[str] = []
    problems += _check_ordering(events)
    cadence_problems, stride = _check_cadence(events)
    problems += cadence_problems
    problems += _check_no_shared_expert(events)

    if stride is not None:
        out.append("re-selection stride: every %d forward passes (constant)" % stride)
    has_xrank = any("xrank" in e for e in events)
    has_basis = any("basis" in e for e in events)
    if not has_xrank:
        out.append(
            "note: no cross-rank data in this journal, so the "
            "shared-expert check could not run."
        )
    if not has_basis:
        out.append(
            "note: no decision basis in this journal, so a diff "
            "cannot separate SCHEDULER from UPSTREAM."
        )
    out.append("")
    if problems:
        out.append("FAIL (%d)" % len(problems))
        out += ["  - " + p for p in problems]
        return False, out
    out.append(
        "OK -- ordered, constant stride, no expert claimed by two ranks at once."
    )
    out.append(
        "This is a self-consistency result, NOT determinism: two runs "
        "can each pass and still differ. Use `diff` for that."
    )
    return True, out


# ---------------------------------------------------------------------------
# Two-run comparison
# ---------------------------------------------------------------------------


def _norm_changes(ev: dict) -> Any:
    xr = ev.get("xrank")
    if xr:
        return {str(k): [list(c) for c in v] for k, v in sorted(xr.items())}
    return [list(c) for c in ev.get("changes", [])]


def _classify(a: dict, b: dict) -> tuple[str, str]:
    """Name the kind of divergence between two events at the same index."""
    fa, fb = a.get("forward_no"), b.get("forward_no")
    da = (a.get("basis") or {}).get("digest")
    db = (b.get("basis") or {}).get("digest")
    ca, cb = _norm_changes(a), _norm_changes(b)

    if fa != fb and ca == cb and da == db:
        return (
            "GATE",
            (
                "Same traffic, same decision, different step: forward %s vs %s. "
                "The trigger moved. Both runs chose correctly and the outputs "
                "still diverge, because the swap lands on a different token -- "
                "this is the signature of a gate that reads something other than "
                "the forward counter." % (fa, fb)
            ),
        )
    if fa == fb and da is not None and da != db:
        return (
            "UPSTREAM",
            (
                "Same step (forward %s), different traffic going in (%s vs %s; "
                "%s vs %s token-expert pairs). The divergence already existed "
                "before the selector saw it -- look at batch composition, "
                "routing, and request arrival, not at the scheduler."
                % (
                    fa,
                    da,
                    db,
                    (a.get("basis") or {}).get("pairs"),
                    (b.get("basis") or {}).get("pairs"),
                )
            ),
        )
    if fa == fb and da == db and ca != cb:
        return (
            "SCHEDULER",
            (
                "Same step (forward %s), identical traffic going in, different "
                "choice coming out. The selector itself is nondeterministic: "
                "given one input it produced two answers." % fa
            ),
        )
    return (
        "MIXED",
        (
            "forward %s vs %s, basis %s vs %s, decisions %s. More than one thing "
            "moved, so the classes above cannot be separated; compare the two "
            "events in full." % (fa, fb, da, db, "differ" if ca != cb else "match")
        ),
    )


def diff(path_a: str, path_b: str) -> tuple[bool, list[str]]:
    """Compare two journals. Returns ``(identical, lines)``."""
    ea, _ = load(path_a)
    eb, _ = load(path_b)
    out = [
        "A: %s (%d events)" % (path_a, len(ea)),
        "B: %s (%d events)" % (path_b, len(eb)),
        "",
    ]

    for i in range(min(len(ea), len(eb))):
        a, b = ea[i], eb[i]
        if (
            a.get("forward_no") == b.get("forward_no")
            and _norm_changes(a) == _norm_changes(b)
            and (a.get("basis") or {}).get("digest")
            == (b.get("basis") or {}).get("digest")
        ):
            continue
        kind, why = _classify(a, b)
        out.append(
            "DIVERGED at event #%d (%s / %s)" % (i, a.get("event"), b.get("event"))
        )
        out.append("")
        out.append("  %s" % kind)
        out.append("  %s" % why)
        out.append("")
        out += _event_detail("  A", a)
        out += _event_detail("  B", b)
        out.append("")
        out.append(
            "Events 0..%d were identical, so this is the FIRST point "
            "the two runs parted company. Everything after it is "
            "consequence, not cause." % (i - 1)
            if i
            else "This is the very first event, so the runs never agreed."
        )
        return False, out

    if len(ea) != len(eb):
        longer, n = ("A", len(ea)) if len(ea) > len(eb) else ("B", len(eb))
        out.append(
            "DIVERGED by length: the first %d events are identical, then %s "
            "continues to %d. One run kept deciding after the other stopped "
            "-- usually a different number of decode steps, so compare the "
            "output lengths before suspecting the scheduler."
            % (min(len(ea), len(eb)), longer, n)
        )
        return False, out

    out.append(
        "IDENTICAL -- %d events, same steps, same traffic, same decisions." % len(ea)
    )
    out.append(
        "The decision sequence is reproducible. Any remaining output "
        "difference is numeric, not scheduling."
    )
    return True, out


def _event_detail(tag: str, ev: dict) -> list[str]:
    basis = ev.get("basis") or {}
    lines = [
        "%s fwd=%s tick=%s src=%s basis=%s pairs=%s"
        % (
            tag,
            ev.get("forward_no"),
            ev.get("tick"),
            ev.get("source"),
            basis.get("digest", "-"),
            basis.get("pairs", "-"),
        )
    ]
    xr = ev.get("xrank")
    if xr:
        for rank_id in sorted(xr, key=lambda x: int(x)):
            ch = xr[rank_id]
            head = ", ".join("L%d:%d>%d" % tuple(c) for c in ch[:4])
            more = "" if len(ch) <= 4 else " ... (%d layers)" % len(ch)
            lines.append("%s   rank%s %s%s" % (tag, rank_id, head, more))
    else:
        ch = ev.get("changes", [])
        head = ", ".join("L%d:%d>%d" % tuple(c) for c in ch[:4])
        more = "" if len(ch) <= 4 else " ... (%d layers)" % len(ch)
        lines.append("%s   %s%s" % (tag, head, more))
    return lines


def main(argv: Optional[list] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m sglang.srt.eplb.cpu_offload.journal",
        description="Check or compare CPU-offload decision journals.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="self-consistency of one journal")
    c.add_argument("journal")
    d = sub.add_parser(
        "diff", help="compare two journals, classify the first divergence"
    )
    d.add_argument("a")
    d.add_argument("b")
    args = p.parse_args(argv)

    try:
        if args.cmd == "check":
            ok, lines = check(args.journal)
        else:
            ok, lines = diff(args.a, args.b)
    except (JournalError, OSError) as e:
        print("error: %s" % e)
        return 2
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
