"""Call-path tracing for the dual-loop CPU-offload MoE path.

WHY THIS EXISTS
---------------
The offload path is assembled from three independently-configured pieces (a
MoE layer class, a dispatch hook, a CUDA-graph offload context), each of which
can silently take a different branch than the reader of the source expects:

  * ``DeepEPMoEHeto`` overrides ``forward`` outright, so the whole of
    ``DeepEPMoE.forward`` -- including its ``deprecate_flag`` early return AND
    a full CPU-offload begin/commit block below it -- is never entered at all.
    Two independent readers of this file concluded instead that the block was
    reached and then skipped by the ``deprecate_flag`` return; the counter
    (parent forward = 0) settled it. Reading the source does not tell you
    which override wins.
  * Both integration points wrap their offload calls in ``try/except`` and
    degrade to GPU-only on failure. That is the right behaviour, but a
    swallowed exception leaves no trace at all: a misconfiguration shows up as
    "performance is wrong" and nothing else.

So this module records which code actually ran, which code never ran, how the
configuration was adjudicated, what each layer decided, and what was swallowed
-- and prints one report at process exit.

ENABLING
--------
``SGLANG_MOE_TRACE=1``. Off by default.

    SGLANG_MOE_TRACE=1              enable
    SGLANG_MOE_TRACE_RANKS=0,4      restrict the printed report to these ranks
                                    (default: every rank that prints METRICS)
    SGLANG_MOE_TRACE_JOURNAL=<path> also stream the decision journal there

THE REPORT AND THE JOURNAL ANSWER DIFFERENT QUESTIONS
-----------------------------------------------------
The exit report is a snapshot plus cumulative counters: it answers "what did
this process do". That is the wrong shape for the defect this module was
written for, which was never visible inside one run -- the two runs each made
a self-consistent set of decisions, and only differed from each other.

So the decision journal is a separate output. It is an append-only record of
every (re-)selection, keyed on the forward-pass counter, holding EVERY rank's
assignment and a fingerprint of the traffic the choice was made from. Two
journals can be compared mechanically, which is what turns "the outputs
differ" into "they parted company at forward 1400, layer 37, and the traffic
going in was identical, so the scheduler is the nondeterministic part".

Three properties it must keep, or it stops being comparable:

  * no wall-clock, no durations, no addresses -- anything that legitimately
    differs between two identical runs is noise that hides the real diff;
  * the forward counter is the only time axis, because after the ownership
    and cadence fixes it is the one quantity guaranteed identical on every
    rank at every step;
  * events are emitted only when something actually changed, so an unchanged
    stretch does not pad the file with rows that always match.

COST WHEN DISABLED
------------------
``ENABLED`` is resolved once, at import. Every call site is written as::

    if _MOE_TRACE:
        _trace.hit(...)

so a disabled run pays one module-global load plus a not-taken branch, and
nothing else -- no function call, no allocation, no import. Call sites inside
``sglang/srt/layers/moe`` additionally defer the *import* of this module until
the first traced call, so a disabled run never even loads it.

The daemon side is gated the same way: the per-layer counters live behind
``SlotHeader::trace_enabled`` (v13+), which the client sets at attach time, so
an untraced run issues zero atomic read-modify-writes for them.

WHAT IS AND IS NOT COUNTED
--------------------------
Python counters count PYTHON EXECUTIONS. Under CUDA graph replay the model's
Python ``forward`` does not run at all -- the recorded kernel sequence does --
so the Python counters reflect capture plus any eager passes, not the decode
steps. The daemon-side per-layer counters do cover every replay. The report
labels both, and never presents one as the other.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import traceback
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Master switch -- resolved once, at import.
# ---------------------------------------------------------------------------

_TRUE = ("1", "true", "yes", "y", "on")


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


ENABLED: bool = _env_on("SGLANG_MOE_TRACE")

# Where to stream the decision journal, if anywhere. Independent of the
# report: a run can want the machine-readable journal and no text, or both.
JOURNAL_PATH: Optional[str] = (
    os.environ.get("SGLANG_MOE_TRACE_JOURNAL", "").strip() or None
)


def _journal_path_for(rank: Any) -> str:
    """Expand the journal path for one rank.

    Tensor parallelism means one process per rank, all of them inheriting the
    same environment. A single literal path would have every rank appending to
    one file: the records interleave, a long one can tear across a write, and
    the ordering the comparison depends on is gone. So ``%r`` in the path is
    replaced with the rank, and a path without it gets a ``.rank<N>`` suffix
    rather than being silently shared.
    """
    tag = "unk" if rank is None else str(rank)
    if "%r" in JOURNAL_PATH:
        return JOURNAL_PATH.replace("%r", tag)
    base, dot, ext = JOURNAL_PATH.rpartition(".")
    if dot:
        return "%s.rank%s.%s" % (base, tag, ext)
    return "%s.rank%s" % (JOURNAL_PATH, tag)


# Bumped whenever the event record changes shape. The comparison tool refuses
# to diff across versions rather than silently matching different fields.
JOURNAL_SCHEMA: int = 1

# Record the full per-layer traffic vector behind each re-selection, not just
# its digest. Off by default because of size, not cost: the numbers have
# already been copied to the host to compute the digest, so keeping them is
# free, but 94 layers x 128 experts is ~48 KB per event per rank and a long
# run has dozens of events on eight ranks.
#
# Worth having despite the size. The journal answers "what was chosen"; the
# digest only answers "was the input the same". Neither can answer why a layer
# keeps flipping between two experts, which is the question an actual run
# raised: 57% of re-selections were a layer returning to the expert it had two
# events ago. Deciding whether that is real traffic movement or the selector
# chasing noise needs the margin between first and second place, and that is
# exactly what the digest throws away.
COUNTS: bool = _env_on("SGLANG_MOE_TRACE_COUNTS")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# Cap on how many events carry full counts. A capture whose length is set by
# generation length should not be able to turn into a multi-gigabyte journal
# by accident; past the cap, events still carry the digest, so the
# determinism comparison is unaffected and only the charts lose their tail.
COUNTS_MAX: int = _env_int("SGLANG_MOE_TRACE_COUNTS_MAX", 60)


def want_counts(event_no: int) -> bool:
    """Whether this re-selection should carry its full traffic vector."""
    return COUNTS and event_no <= COUNTS_MAX


# Which ranks print the report. Empty = all of them.
_REPORT_RANKS: Optional[set] = None
_raw_ranks = os.environ.get("SGLANG_MOE_TRACE_RANKS", "").strip()
if _raw_ranks:
    try:
        _REPORT_RANKS = {int(x) for x in _raw_ranks.split(",") if x.strip()}
    except ValueError:
        _REPORT_RANKS = None


# ---------------------------------------------------------------------------
# The path map.
#
# Declared statically, ALL OF IT, so a node that never executed still has a row
# in the report. A report that only listed what ran would let a reader conclude
# "this is the path" without ever learning that a second path existed and was
# never taken -- which is the specific failure this file was written for.
#
# (key, depth, label, note)
# ---------------------------------------------------------------------------

_PATH_SPECS: list[tuple[str, int, str, str]] = [
    # -- layer entry points -------------------------------------------------
    # DeepEPMoEHeto OVERRIDES forward and reaches FusedMoE.forward directly,
    # so DeepEPMoE.forward (the parent, with its own offload block) is a
    # separate node that may well never run. Collapsing the two would let a
    # zero on one be read as a zero on the other.
    ("moe.heto.forward", 0, "DeepEPMoEHeto.forward", ""),
    ("moe.heto.gpu_only", 1, "cpu_moe is None -> _forward_gpu_experts()", ""),
    ("moe.heto.decode_gate", 1, "decode-only gate -> _forward_gpu_experts()", ""),
    ("moe.heto.hetero", 1, "in-class hetero path (GPU + self.cpu_moe)", ""),
    ("moe.heto.hetero_mask", 2, "-> in-class cpu_slot mask applied", ""),
    (
        "moe.heto.gpu_experts",
        1,
        "_forward_gpu_experts -> FusedMoE.forward(self, ...)",
        "",
    ),
    ("moe.deepep.forward", 0, "DeepEPMoE.forward (parent class)", ""),
    ("moe.deepep.super", 1, "deprecate_flag=True -> super().forward()", ""),
    ("moe.deepep.inclass_begin", 1, "in-class offload begin (ep_moe/layer.py)", ""),
    ("moe.deepep.inclass_armed", 2, "-> ctx returned (topk masked)", ""),
    ("moe.deepep.inclass_commit", 1, "in-class offload commit", ""),
    ("moe.fused.forward", 0, "FusedMoE.forward", ""),
    ("moe.fused.begin", 1, "_maybe_begin_cpu_offload", ""),
    ("moe.fused.begin_armed", 2, "-> ctx returned (topk masked)", ""),
    ("moe.fused.begin_passthrough", 2, "-> None (pure pass-through)", ""),
    ("moe.fused.commit", 1, "_commit_cpu_offload", ""),
    # -- shared dispatch hook ----------------------------------------------
    ("hook.begin", 0, "begin_cpu_offload (dispatch_hook.py)", ""),
    ("hook.begin.capture", 1, "CUDA-graph CAPTURE branch", ""),
    ("hook.begin.capture_armed", 2, "-> gctx.begin_offload returned ctx", ""),
    ("hook.begin.capture_skip", 2, "-> skipped", "reason"),
    ("hook.begin.graph_suppressed", 1, "graph ctx live -> eager path suppressed", ""),
    ("hook.begin.eager", 1, "eager branch", ""),
    ("hook.begin.eager_armed", 2, "-> daemon request submitted", ""),
    ("hook.begin.eager_skip", 2, "-> skipped", "reason"),
    ("hook.commit", 0, "commit_cpu_offload (dispatch_hook.py)", ""),
    ("hook.commit.graph", 1, "CUDA-graph path (gctx.commit_offload)", ""),
    ("hook.commit.eager", 1, "eager path (wait + index_add)", ""),
    ("hook.commit.timeout", 1, "wait-kernel TIMEOUT -> CPU contribution discarded", ""),
    ("hook.commit.nan", 1, "NaN/Inf in CPU output -> zeroed", ""),
    # -- CUDA-graph offload context ----------------------------------------
    ("gctx.begin", 0, "CudaGraphOffloadContext.begin_offload", ""),
    (
        "gctx.begin.fused_prepare",
        1,
        "fused prepare kernel (CPU_OFFLOAD_FUSED_PREPARE=1)",
        "",
    ),
    (
        "gctx.begin.fused_prepare_fallback",
        1,
        "-> fused prepare unusable, original small ops ran",
        "reason",
    ),
    ("gctx.begin.ok", 1, "-> submitted (compact gather + D2H + submit)", ""),
    ("gctx.begin.reject", 1, "-> returned None", "reason"),
    ("gctx.begin.prefetch", 1, "next-layer prefetch submitted", ""),
    ("gctx.commit", 0, "CudaGraphOffloadContext.commit_offload", ""),
    (
        "gctx.commit.fused_add",
        1,
        "combine path A: scatter_add_compact_status (fused)",
        "",
    ),
    ("gctx.commit.fallback", 1, "combine path B: scatter_compact + mask + add", ""),
    # -- fast loop ----------------------------------------------------------
    ("loop.tick", 0, "FastLoopScheduler.tick", ""),
    ("loop.bootstrap", 1, "bootstrap slot install (first tick)", ""),
    ("loop.run_once", 1, "_run_once (re-selection ran)", ""),
    ("loop.run_once.no_data", 2, "-> bailed out, no traffic evidence", ""),
    (
        "loop.run_once.pinned",
        2,
        "-> bailed out, CPU_OFFLOAD_USE_LAST_GPU_SLOT pins the layout",
        "",
    ),
    ("loop.xrank_sync", 1, "cross-rank all_reduce of expert traffic", ""),
]

_PATH_ORDER = [k for k, _, _, _ in _PATH_SPECS]
_PATH_META = {k: (d, label, note) for k, d, label, note in _PATH_SPECS}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class _State:
    """All trace state for this process. One instance, module-global."""

    __slots__ = (
        "counts",
        "reasons",
        "facts",
        "fact_order",
        "exceptions",
        "reselects",
        "layer_py",
        "server_args",
        "lock",
        "reported",
        "module_local_experts",
        "block_rank",
        "block_num_local",
        "journal_fh",
        "journal_broken",
    )

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        # key -> {reason: count}
        self.reasons: dict[str, dict[str, int]] = {}
        # name -> (value, source)
        self.facts: dict[str, tuple[Any, str]] = {}
        self.fact_order: list[str] = []
        # site -> {"count": n, "type": str, "msg": str, "tb": str}
        self.exceptions: dict[str, dict] = {}
        self.reselects: list[dict] = []
        # layer_id -> {counter: n}
        self.layer_py: dict[int, dict[str, int]] = {}
        # layer_id -> set of global expert ids this rank's GPU actually
        # computes, taken from the MoE module's own dispatch map.
        self.module_local_experts: dict[int, set] = {}
        # The parameters of the ONE mapping that actually decides ownership:
        # rank r owns global expert ids [r*n, (r+1)*n). Both the weight loader
        # (FusedMoE._map_global_expert_id_to_local_expert_id) and the
        # dispatcher (StandardDispatcher.dispatch) compute exactly this.
        self.block_rank: Optional[int] = None
        self.block_num_local: Optional[int] = None
        self.server_args: Any = None
        self.lock = threading.Lock()
        self.reported = False
        # Journal sink. Opened lazily on the first event so a run that never
        # re-selects leaves no stray empty file.
        self.journal_fh: Any = None
        self.journal_broken = False


_S = _State()


# ---------------------------------------------------------------------------
# Recording API. Every one of these is called only from behind an
# ``if _MOE_TRACE:`` guard, so none of them needs its own fast path.
# ---------------------------------------------------------------------------


def hit(key: str, n: int = 1) -> None:
    """Count one execution of a declared code path."""
    _S.counts[key] = _S.counts.get(key, 0) + n


def reason(key: str, why: str, n: int = 1) -> None:
    """Count one execution of ``key`` and attribute it to a reason."""
    _S.counts[key] = _S.counts.get(key, 0) + n
    bucket = _S.reasons.setdefault(key, {})
    bucket[why] = bucket.get(why, 0) + n


def layer_hit(layer_id: int, key: str, n: int = 1) -> None:
    """Count a per-layer event (Python-side)."""
    row = _S.layer_py.setdefault(int(layer_id), {})
    row[key] = row.get(key, 0) + n


def fact(name: str, value: Any, source: str) -> None:
    """Record one adjudicated configuration value and where it came from.

    ``source`` is free text but should name the channel, e.g.
    ``env:CPU_OFFLOAD_MAX_SPIN``, ``server_args:--cpu-offload-max-spin``,
    ``default``, or ``runtime`` for something only observable once the model
    is built. Later calls overwrite earlier ones (last writer wins) so a
    runtime observation can replace a boot-time guess.
    """
    if name not in _S.facts:
        _S.fact_order.append(name)
    _S.facts[name] = (value, source)


def exc(site: str, exception: BaseException) -> None:
    """Record an exception that the caller is about to swallow.

    The first occurrence keeps its full traceback; later ones only bump the
    count. Aggregating by site is what makes this readable -- a swallowed
    exception on a per-layer path fires tens of thousands of times.
    """
    entry = _S.exceptions.get(site)
    if entry is None:
        _S.exceptions[site] = {
            "count": 1,
            "type": type(exception).__name__,
            "msg": str(exception)[:400],
            "tb": "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )[-4000:],
        }
    else:
        entry["count"] += 1


def reselect(
    *,
    event: str,
    tick: int,
    forward_no: int,
    changes: list,
    source: str = "",
    xrank: Optional[dict] = None,
    basis: Optional[dict] = None,
    rank: Optional[int] = None,
) -> None:
    """Record one CPU-slot (re-)selection event.

    ``changes`` is a list of ``(layer_id, old_expert, new_expert)`` for the
    LOCAL rank. Only genuine changes belong here -- a tick that decided
    nothing new should not produce an event, or the history stops being
    diffable between two runs.

    ``xrank`` is the same thing for EVERY rank (``rank_id -> [(layer, old,
    new), ...]``). The local list alone cannot answer the question invariant 2
    actually turns on, which is whether two ranks both targeted one expert:
    that is a property of the whole assignment matrix, and each rank's report
    only ever saw its own column of it.

    ``basis`` fingerprints what the choice was made FROM -- the per-layer
    traffic the selector ranks on -- as ``{"layers": n, "pairs": n, "digest":
    str}``. Without it a mismatch between two runs says only "they differ";
    with it, identical basis and different outcome isolates the scheduler,
    while a different basis sends you upstream to batching and routing.
    """
    ev = {
        "event": event,
        "tick": int(tick),
        "forward_no": int(forward_no),
        "source": source,
        "changes": sorted((int(a), int(b), int(c)) for a, b, c in changes),
    }
    if rank is not None:
        ev["rank"] = int(rank)
    if xrank:
        ev["xrank"] = {
            int(r): sorted((int(a), int(b), int(c)) for a, b, c in ch)
            for r, ch in xrank.items()
        }
    if basis:
        ev["basis"] = dict(basis)
    _S.reselects.append(ev)
    _journal_write(ev)


def _journal_write(ev: dict) -> None:
    """Append one event to the journal file, if one was configured.

    Written and flushed as it happens rather than at exit: a run that hangs or
    is killed is exactly the run whose decision history is worth having, and
    an atexit-only sink loses precisely those. One line of JSON per event, so
    a partially-written file is still parseable up to the truncation.
    """
    if not JOURNAL_PATH or _S.journal_broken:
        return
    try:
        if _S.journal_fh is None:
            path = _journal_path_for(ev.get("rank"))
            # "w", not "a": appending would silently glue a new run onto a
            # previous one, and the result would compare as a long run that
            # diverged in the middle -- a fabricated verdict.
            _S.journal_fh = open(path, "w", encoding="utf-8")
            _S.journal_fh.write(
                json.dumps({"event": "_schema", "v": JOURNAL_SCHEMA}) + "\n"
            )
        # sort_keys so two runs produce byte-comparable lines, and no
        # timestamp anywhere: a field that legitimately differs every run
        # would swamp the diff this file exists to support.
        _S.journal_fh.write(json.dumps(ev, sort_keys=True) + "\n")
        _S.journal_fh.flush()
    except Exception as e:  # pragma: no cover - diagnostic sink only
        # A broken sink must never take the model down, and must never
        # silently look like "this run made no decisions".
        _S.journal_broken = True
        exc("trace._journal_write", e)


def record_server_args(server_args: Any) -> None:
    _S.server_args = server_args


# ---------------------------------------------------------------------------
# Configuration provenance
#
# Deliberately re-derived here rather than instrumented into
# load_cpu_offload_config(): that function is one large expression whose
# evaluation order IS the precedence rule, and rewriting it to thread a
# provenance argument through would risk changing behaviour for an
# observability feature. The table below restates the same precedence
# (explicit ServerArgs value > environment variable > dataclass default) and
# is checked against the resolved config -- a mismatch is reported rather than
# hidden, so the two can never drift apart silently.
# ---------------------------------------------------------------------------

# (config field, ServerArgs attribute, env var, human label)
_CONFIG_CHANNELS: list[tuple[str, Optional[str], Optional[str], str]] = [
    (
        "enable_cpu_offload_fast_loop",
        "enable_cpu_offload_fast_loop",
        "CPU_OFFLOAD_ENABLE_FAST_LOOP",
        "fast loop enabled",
    ),
    ("cpu_active_ranks", "cpu_active_ranks", "CPU_ACTIVE_RANKS", "active ranks"),
    ("cpu_offload_backend", "cpu_offload_backend", "CPU_OFFLOAD_BACKEND", "backend"),
    ("cpu_offload_decode_only", "cpu_offload_decode_only", None, "decode-only gate"),
    (
        "cpu_offload_top_m",
        "cpu_offload_top_m",
        "CPU_OFFLOAD_TOP_M",
        "slots per rank (top_m)",
    ),
    # The ms knob is now only an on/off switch (0 == never re-select); the
    # cadence comes from the forward-pass knob below. Labelled so the table
    # cannot be read as "re-selects every 500 ms".
    (
        "cpu_offload_slot_update_interval_ms",
        "cpu_offload_slot_update_interval_ms",
        "CPU_OFFLOAD_SLOT_UPDATE_INTERVAL_MS",
        "re-select ENABLED (0=off, ms ignored)",
    ),
    (
        "cpu_offload_slot_update_every_n_forwards",
        None,
        "CPU_OFFLOAD_SLOT_UPDATE_EVERY_N_FORWARDS",
        "re-select every N forward passes",
    ),
    (
        "cpu_offload_xrank_sync_every_n_ticks",
        None,
        "CPU_OFFLOAD_XRANK_SYNC_EVERY_N_TICKS",
        "cross-rank sync every N ticks",
    ),
    (
        "cpu_offload_max_spin",
        "cpu_offload_max_spin",
        "CPU_OFFLOAD_MAX_SPIN",
        "wait-kernel max spins",
    ),
    (
        "cpu_offload_timeout_us",
        "cpu_offload_timeout_us",
        "CPU_OFFLOAD_TIMEOUT_US",
        "timeout (us)",
    ),
    (
        "cpu_offload_enable_mode_a",
        "cpu_offload_enable_mode_a",
        "CPU_OFFLOAD_ENABLE_MODE_A",
        "Mode A (token split)",
    ),
    (
        "cpu_offload_shm_prefix",
        "cpu_offload_shm_prefix",
        "CPU_OFFLOAD_SHM_PREFIX",
        "SHM prefix",
    ),
    (
        "cpu_offload_num_workers_per_numa",
        "cpu_offload_num_workers_per_numa",
        "CPU_OFFLOAD_NUM_WORKERS_PER_NUMA",
        "workers per NUMA",
    ),
    (
        "cpu_offload_worker_assign_strategy",
        "cpu_offload_worker_assign_strategy",
        "CPU_OFFLOAD_WORKER_ASSIGN_STRATEGY",
        "rank->worker strategy",
    ),
    (
        "cpu_offload_enable_cuda_graph_polling",
        "cpu_offload_enable_cuda_graph_polling",
        "CPU_OFFLOAD_ENABLE_CUDA_GRAPH_POLLING",
        "CUDA-graph polling",
    ),
]

# Environment-only knobs that never had a ServerArgs channel but do change the
# path. Listed separately so the report shows them with their real default
# instead of pretending they were adjudicated.
_ENV_ONLY_KNOBS: list[tuple[str, str, str]] = [
    ("SGLANG_ENABLE_JIT_DEEPGEMM", "0", "DeepGEMM JIT (drives deprecate_flag)"),
    ("CPU_OFFLOAD_GPU_QUANT", "1", "GPU-side int8 activation quant"),
    ("CPU_OFFLOAD_MAX_HITS", "", "max CPU rows per layer call"),
    ("CPU_OFFLOAD_METRICS_DETAIL", "0", "extra metric kernels per layer"),
    ("CPU_OFFLOAD_PREFETCH_NEXT_LAYER", "1", "next-layer weight prewarm"),
    ("CPU_OFFLOAD_USE_LAST_GPU_SLOT", "0", "pin cpu_slot to own last GPU expert"),
    ("CPU_OFFLOAD_FORCE_NO_SLOT", "0", "force cpu_slot=-1 (offload wired, idle)"),
    ("SGLANG_CPU_OFFLOAD_INIT_SLOT", "", "bootstrap slot override (flat)"),
    ("SGLANG_CPU_OFFLOAD_INIT_SLOTS", "", "bootstrap slot override (per rank)"),
    ("SGLANG_CPU_OFFLOAD_RECORD_LOAD", "0", "python-side per-expert load counter"),
    ("MOE_DTYPE", "", "CPU engine compute dtype"),
    ("CPU_OFFLOAD_DAEMON_LOADS_WEIGHTS", "0", "daemon loads full E from disk"),
]


def _serverargs_channel_is_wired(field: str) -> bool:
    """Is there actually a path from ``ServerArgs`` to this config field?

    A knob can exist on ``ServerArgs`` (and be accepted on the command line)
    while nothing ever forwards it into ``CpuOffloadConfig`` -- in which case
    the CLI flag is inert and the value silently comes from the dataclass
    default. Reporting such a value as "came from ServerArgs" would be exactly
    the kind of confident wrong answer this report exists to prevent, so the
    claim is checked instead of assumed: the field must both be a parameter of
    ``load_cpu_offload_config`` AND be passed by
    ``_build_config_from_server_args``. Read-only introspection; on any doubt
    this returns True and the row falls back to the ordinary precedence rule.
    """
    try:
        import inspect
        import re

        from sglang.srt.eplb.cpu_offload import config as _cfgmod
        from sglang.srt.eplb.cpu_offload import runtime as _rtmod

        if field not in inspect.signature(_cfgmod.load_cpu_offload_config).parameters:
            return False
        src = inspect.getsource(_rtmod._build_config_from_server_args)
        return re.search(r"\b%s\s*=" % re.escape(field), src) is not None
    except Exception:
        return True


def _provenance_rows(cfg: Any) -> list[tuple[str, str, str, str]]:
    """Return ``(label, field, value, source)`` for every adjudicated knob."""
    sa = _S.server_args
    rows: list[tuple[str, str, str, str]] = []
    for field, sa_attr, env_name, label in _CONFIG_CHANNELS:
        value = getattr(cfg, field, "<absent>")
        source = "default"
        wired = _serverargs_channel_is_wired(field) if sa_attr else True
        if sa_attr is not None and sa is not None and wired:
            sa_val = getattr(sa, sa_attr, None)
            if sa_val is not None:
                source = f"ServerArgs.{sa_attr}"
        if source == "default" and env_name and env_name in os.environ:
            source = f"env:{env_name}"
        if sa_attr is not None and not wired:
            # The flag parses, but nothing carries it into CpuOffloadConfig.
            source += "  !! --%s is INERT (not forwarded)" % sa_attr.replace("_", "-")
        rows.append((label, field, _fmt(value), source))
    return rows


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "-"
    s = str(v)
    # The report is 100 columns wide and the fact column starts at 37, so a
    # value has 60 to spare before wrapping; keep the whole line on one row.
    return s if len(s) <= 62 else s[:59] + "..."


# ---------------------------------------------------------------------------
# Runtime facts that are only observable once the model exists.
# ---------------------------------------------------------------------------


def record_local_expert_ids(layer_id: int, ids) -> None:
    """Record which global expert ids this rank's GPU computes for a layer.

    Taken from the MoE module's OWN dispatch map (``_global_to_local_map``),
    which is what actually decides whether a routed pair is computed on this
    rank -- not from the EPLB metadata, which the offload code paths read
    instead and which can describe a different placement (see
    ``_placement_summary``).
    """
    try:
        _S.module_local_experts[int(layer_id)] = {int(x) for x in ids}
    except Exception:
        pass


def record_moe_layer_facts(module: Any) -> None:
    """Capture the static routing decisions of one MoE layer.

    Called once, from the first MoE layer's ``forward``. Everything recorded
    here is fixed at construction time, so recording it from the first call
    costs one branch per subsequent call and nothing else.
    """
    try:
        fact("moe class", type(module).__name__, "runtime")
        dep = getattr(module, "deprecate_flag", None)
        if dep is None:
            fact("forward branch", "no deprecate_flag on this class", "runtime")
        elif dep:
            fact(
                "forward branch",
                "deprecate_flag=True -> FusedMoE.forward (in-class hook DEAD)",
                "runtime",
            )
        else:
            fact(
                "forward branch",
                "deprecate_flag=False -> in-class SBO path",
                "runtime",
            )
        quant = getattr(module, "quant_config", None)
        fact(
            "deprecate_flag basis",
            "quant_config=%s  SGLANG_ENABLE_JIT_DEEPGEMM=%s"
            % (
                type(quant).__name__ if quant is not None else "None",
                os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "<unset>"),
            ),
            "runtime",
        )
        disp = getattr(module, "dispatcher", None)
        fact(
            "dispatcher",
            type(disp).__name__ if disp is not None else "<none>",
            "runtime",
        )
        fact(
            "reduction",
            "reduce_results=%s moe_tp_size=%s moe_ep_size=%s"
            % (
                getattr(module, "reduce_results", "?"),
                getattr(module, "moe_tp_size", "?"),
                getattr(module, "moe_ep_size", "?"),
            ),
            "runtime",
        )
        fact(
            "expert counts",
            "num_experts=%s num_local_experts=%s top_k=%s"
            % (
                getattr(module, "num_experts", "?"),
                getattr(module, "num_local_experts", "?"),
                getattr(getattr(module, "moe_runner_config", None), "top_k", "?"),
            ),
            "runtime",
        )
        qm = getattr(module, "quant_method", None)
        fact(
            "quant method",
            type(qm).__name__ if qm is not None else "<none>",
            "runtime",
        )
        # THE mapping that decides ownership. Two independent places compute
        # it, both from (moe_ep_rank, num_local_experts) and both contiguous:
        #   FusedMoE._map_global_expert_id_to_local_expert_id -- decides which
        #     expert's weights get loaded into which local slot;
        #   StandardDispatcher.dispatch -- builds local_expert_mapping, so a
        #     routed pair outside the block becomes -1 and is not computed.
        # So "does this rank compute expert e" == "e // n == rank".
        try:
            _r = int(getattr(module, "moe_ep_rank"))
            _n = int(getattr(module, "num_local_experts"))
            if _n > 0:
                _S.block_rank, _S.block_num_local = _r, _n
                fact(
                    "gpu experts (AUTHORITATIVE, block)",
                    "moe_ep_rank=%d num_local_experts=%d -> this rank owns "
                    "[%d, %d)" % (_r, _n, _r * _n, (_r + 1) * _n),
                    "runtime",
                )
        except Exception:
            pass
        # The module ALSO builds an interleaved _global_to_local_map from its
        # own expert_map. Recorded only so the report can show that it
        # disagrees with the block layout AND that nothing reads it.
        g2l = getattr(module, "_global_to_local_map", None)
        if isinstance(g2l, dict) and g2l:
            ids = sorted(int(k) for k in g2l)
            lid = getattr(module, "layer_id", None)
            if isinstance(lid, int):
                record_local_expert_ids(lid, ids)
            stride = (ids[1] - ids[0]) if len(ids) > 1 else 0
            contiguous = (
                all(b - a == 1 for a, b in zip(ids, ids[1:])) if len(ids) > 1 else True
            )
            fact(
                "gpu experts (_global_to_local_map, DEAD)",
                "n=%d layout=%s ids=[%s ... %s]"
                % (
                    len(ids),
                    "block/contiguous"
                    if contiguous
                    else "interleaved stride=%d" % stride,
                    ", ".join(str(x) for x in ids[:5]),
                    ", ".join(str(x) for x in ids[-3:]),
                ),
                "runtime",
            )
    except Exception:  # never let tracing break the model
        pass


def record_graph_facts(gctx: Any) -> None:
    """Capture the CUDA-graph offload context's shape.

    Called again on every layer registration, not only at initialize(), so
    the recorded layer count is the final one rather than the zero it always
    is while the context is still being built.
    """
    try:
        fact(
            "cuda graph offload",
            "initialized  layers=%d max_tokens=%d max_hits=%d gpu_quant=%s"
            % (
                len(getattr(gctx, "_layers", {}) or {}),
                getattr(gctx, "max_tokens", -1),
                getattr(gctx, "max_hits", -1),
                getattr(gctx, "_gpu_quant", "?"),
            ),
            "runtime",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Expert placement
# ---------------------------------------------------------------------------


def _local_expert_sets(local_rank: int, num_layers: int) -> dict:
    """``layer_id -> set of logical expert ids this rank holds on its GPU``.

    Read from the EPLB metadata rather than from any offload-side bookkeeping.
    Be precise about what that buys: this is independent of the offload
    layer's own records, but it is the SAME metadata the fast-loop candidate
    filter reads, so it cannot contradict the selector's notion of ownership
    -- the two agree by construction. The genuinely independent answer is the
    contiguous block range observed on the MoE module itself
    (``_S.block_rank``), which is what the report prefers. This function is
    the fallback for when that module was never seen, and one of the three
    inputs to ``_placement_disagreement``.
    """
    out: dict[int, set] = {}
    try:
        from sglang.srt.eplb.expert_location import (
            get_global_expert_location_metadata,
        )

        meta = get_global_expert_location_metadata()
        if meta is None:
            return out
        num_local = int(meta.num_local_physical_experts)
        phys = meta.physical_to_logical_map
        lo = local_rank * num_local
        hi = lo + num_local
        rows = int(phys.shape[0])
        for layer_id in range(min(num_layers, rows)):
            row = phys[layer_id, lo:hi]
            out[layer_id] = {int(x) for x in row.tolist() if int(x) >= 0}
    except Exception:
        pass
    return out


def _placement_summary(local_rank: int) -> str:
    try:
        from sglang.srt.eplb.expert_location import (
            get_global_expert_location_metadata,
        )

        meta = get_global_expert_location_metadata()
        if meta is None:
            return "EPLB metadata unavailable"
        num_local = int(meta.num_local_physical_experts)
        phys = meta.physical_to_logical_map
        row = phys[0].tolist()
        lo = local_rank * num_local
        mine = [int(x) for x in row[lo : lo + num_local]]
        contiguous = (
            mine == list(range(mine[0], mine[0] + len(mine))) if mine else False
        )
        layout = "block/contiguous" if contiguous else "strided or remapped"
        head = ", ".join(str(x) for x in mine[:6])
        tail = ", ".join(str(x) for x in mine[-3:])
        return "num_local_physical_experts=%d  layout=%s  layer0 mine=[%s ... %s]" % (
            num_local,
            layout,
            head,
            tail,
        )
    except Exception as e:  # pragma: no cover - diagnostic only
        return "unavailable (%s)" % e


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_W = 100


def _rule(ch: str = "-") -> str:
    return ch * _W


def _section(n: str, title: str) -> str:
    return "\n%s\n%s %s\n%s" % (_rule("="), n, title, _rule("="))


def _tree_lines() -> list[str]:
    lines: list[str] = []
    for key in _PATH_ORDER:
        depth, label, note = _PATH_META[key]
        n = _S.counts.get(key, 0)
        ran = n > 0
        if depth == 0:
            prefix = ""
            glyph = "  " if ran else "X "
        else:
            prefix = "  " * depth
            glyph = "|-> " if ran else "X   "
        text = "%s%s%s" % (prefix, glyph, label)
        lines.append("%-80s %12d" % (text[:80], n))
        if note == "reason":
            for why, cnt in sorted(
                _S.reasons.get(key, {}).items(), key=lambda kv: -kv[1]
            ):
                lines.append(
                    "%-80s %12d" % (("%s      reason: %s" % (prefix, why))[:80], cnt)
                )
    return lines


def _daemon_layer_stats(rt: Any) -> dict:
    """Per-layer daemon counters, or ``{}`` if unavailable."""
    try:
        return rt.engine.read_layer_stats() or {}
    except Exception:
        return {}


def build_report(rt: Any) -> str:
    """Render the whole report as a string. Safe to call more than once."""
    local_rank = getattr(rt, "local_rank", -1)
    cfg = getattr(rt, "config", None)

    out: list[str] = []
    a = out.append

    a("")
    a(_rule("="))
    a("MOE CALL-PATH TRACE   rank=%s   (SGLANG_MOE_TRACE=1)" % local_rank)
    a(_rule("="))

    # ---- ABI ---------------------------------------------------------
    client_ver, daemon_ver = _abi_versions(rt)
    a(
        "slot ABI: client build v%s   daemon slot file v%s   %s"
        % (
            client_ver,
            daemon_ver,
            "OK" if client_ver == daemon_ver else "*** MISMATCH ***",
        )
    )

    # Point at the journal. Everything below describes ONE run, and the
    # defect this tooling exists for was invisible in one run: both runs
    # produced an internally consistent report and differed only from each
    # other. Someone reading a clean report should know where the comparison
    # lives instead of concluding there is nothing left to check.
    if JOURNAL_PATH:
        a("decision journal: %s  (compare two runs with:" % JOURNAL_PATH)
        a("  python -m sglang.srt.eplb.cpu_offload.journal diff A.jsonl B.jsonl)")
    else:
        a("decision journal: off. This report covers ONE run and cannot show")
        a("  a run-to-run difference. Set SGLANG_MOE_TRACE_JOURNAL=<path> on")
        a("  two runs, then: python -m sglang.srt.eplb.cpu_offload.journal")
        a("  diff A.jsonl B.jsonl")

    # ---- 1. paths ----------------------------------------------------
    a(_section("1.", "CODE PATHS EXECUTED  (X = never executed)"))
    a("Python counters count PYTHON executions. Under CUDA-graph replay the")
    a("model's Python forward does not run, so decode steps appear here only")
    a("via capture; section 3's daemon columns cover every replay.")
    a(_rule())
    a("%-80s %12s" % ("path", "count"))
    a(_rule())
    for line in _tree_lines():
        a(line)

    # ---- 2. configuration --------------------------------------------
    a(_section("2.", "HOW THE CONFIGURATION ADJUDICATED THIS PATH"))
    if cfg is not None:
        a("%-36s %-26s %s" % ("knob", "effective value", "came from"))
        a(_rule())
        for label, _field, value, source in _provenance_rows(cfg):
            a("%-36s %-26s %s" % (label[:36], value[:26], source))
    a("")
    a("environment-only knobs (no ServerArgs channel):")
    a(_rule())
    for name, default, label in _ENV_ONLY_KNOBS:
        raw = os.environ.get(name)
        if raw is None:
            a("%-36s %-26s %s" % (name[:36], "(unset -> %s)" % (default or "-"), label))
        else:
            a("%-36s %-26s %s" % (name[:36], raw[:26], label))
    a("")
    a("observed at runtime:")
    a(_rule())
    for name in _S.fact_order:
        value, source = _S.facts[name]
        a("%-36s %s" % (name[:36], _fmt(value)))
    a("%-36s %s" % ("gpu expert placement", _placement_summary(local_rank)))

    # ---- 3. per-layer ------------------------------------------------
    a(_section("3.", "PER-LAYER OFFLOAD DECISION"))
    try:
        _active = bool(rt.is_active_locally())
    except Exception:
        _active = True
    if not _active:
        a("This rank is NOT in cpu_active_ranks, so it never offloads: every")
        a("row below is the pass-through case. That is the expected state for")
        a("a rank the configuration left out, not a fault.")
        a("")
    a(_per_layer_table(rt, local_rank))

    # ---- 4. re-selection ---------------------------------------------
    a(_section("4.", "DYNAMIC RE-SELECTION HISTORY"))
    a(_reselect_report())

    # ---- 5. swallowed exceptions -------------------------------------
    a(_section("5.", "SWALLOWED EXCEPTIONS"))
    if not _S.exceptions:
        a("none recorded.")
    else:
        for site, e in sorted(_S.exceptions.items(), key=lambda kv: -kv[1]["count"]):
            a("")
            a("%s   x%d   %s: %s" % (site, e["count"], e["type"], e["msg"]))
            a("  first traceback:")
            for tl in e["tb"].rstrip().splitlines():
                a("    " + tl)

    a("")
    a(_rule("="))
    a("END MOE CALL-PATH TRACE   rank=%s" % local_rank)
    a(_rule("="))
    return "\n".join(out)


def _abi_versions(rt: Any) -> tuple[str, str]:
    client = "?"
    daemon = "?"
    try:
        import moe_cpu_engine as _e

        if hasattr(_e, "cpu_client_slot_abi_version"):
            client = str(_e.cpu_client_slot_abi_version())
    except Exception:
        pass
    try:
        stats = rt.engine.read_layer_stats() or {}
        if stats.get("version") is not None:
            daemon = str(stats["version"])
    except Exception:
        pass
    return client, daemon


def _per_layer_table(rt: Any, local_rank: int) -> str:
    dstats = _daemon_layer_stats(rt)
    dlayers = dstats.get("layers", {}) or {}
    layer_ids = set(int(k) for k in dlayers.keys()) | set(_S.layer_py.keys())
    try:
        layer_ids |= set(int(x) for x in rt.engine.registered_layers())
    except Exception:
        pass
    if not layer_ids:
        return "no MoE layer was ever registered with the CPU offload engine."

    max_layer = max(layer_ids)
    eplb_sets = _local_expert_sets(local_rank, max_layer + 1)
    module_sets = _S.module_local_experts
    module_any = next(iter(module_sets.values()), None)
    # Ownership comes from the block formula -- see _S.block_rank. Fall back
    # to the EPLB slice only if the module was never observed, and say so.
    blk_r, blk_n = _S.block_rank, _S.block_num_local
    blk_lo = blk_r * blk_n if (blk_r is not None and blk_n) else None
    blk_hi = blk_lo + blk_n if blk_lo is not None else None
    # Which source the 'own?' column below is actually computed from. The
    # legend and the summary line are written from this, because a report
    # that names a source it did not use is worse than one that says
    # "unknown" -- that is exactly how the original 94/94-correct
    # misdiagnosis survived as long as it did.
    own_src = "block" if blk_lo is not None else ("eplb" if eplb_sets else "none")

    lines: list[str] = []
    lines.append("cpu_slot = the expert this rank hands to the CPU for that layer.")
    lines.append("own?     = is that expert one of THIS rank's own GPU experts?")
    if own_src == "block":
        lines.append(
            "           Decided by the CONTIGUOUS BLOCK rank owns [r*n, (r+1)*n),"
        )
        lines.append("           which is what BOTH the weight loader")
        lines.append(
            "           (FusedMoE._map_global_expert_id_to_local_expert_id) and"
        )
        lines.append("           the dispatcher (StandardDispatcher.dispatch) compute.")
    elif own_src == "eplb":
        lines.append("           FALLBACK -- the MoE module was never observed, so the")
        lines.append(
            "           authoritative block range is unknown and this column is"
        )
        lines.append("           computed from the EPLB physical_to_logical_map slice.")
        lines.append(
            "           That is the SAME metadata the fast-loop selector reads,"
        )
        lines.append(
            "           so it is NOT an independent check here -- it agrees with"
        )
        lines.append(
            "           the selector by construction. Treat 'yes' as unverified,"
        )
        lines.append("           and see the cross-check at the end of this section.")
    else:
        lines.append("           UNAVAILABLE -- neither the MoE module nor the EPLB")
        lines.append("           metadata was observed, so every row below shows '?'.")
    lines.append("   yes -> masking it to -1 really does remove the GPU copy, so the")
    lines.append("          pair is computed exactly once. This is the intended state.")
    lines.append("   NO  -> the mask is a no-op here: this rank was never going to")
    lines.append("          compute that expert. The rank that OWNS it still computes")
    lines.append("          it on GPU, and this rank adds a CPU result on top, so the")
    lines.append("          pair is counted TWICE after the TP all-reduce.")
    lines.append("          Reported here, not repaired.")
    lines.append("py_*     = Python-side (capture + eager). dmn_* covers every replay.")
    lines.append(
        "mask     = OK once the cpu_slot branch was masked out of the GPU topk"
    )
    lines.append("           for this layer (i.e. begin returned a ctx and the caller")
    lines.append(
        "           substituted masked_topk_ids); NEVER means it never was, so"
    )
    lines.append("           the GPU still computes that expert too.")
    lines.append(_rule())
    lines.append(
        "%5s %8s %5s %6s %8s %8s %9s %9s %10s %10s %8s %8s"
        % (
            "layer",
            "cpu_slot",
            "own?",
            "mask",
            "py_call",
            "py_armed",
            "dmn_call",
            "dmn_hit",
            "hit_toks",
            "pairs",
            "avg_us",
            "max_us",
        )
    )
    lines.append(_rule())

    own_yes = 0
    own_no = 0
    n_rows = 0
    tot_hit_toks = 0
    for layer_id in sorted(layer_ids):
        slot = -1
        try:
            slot = int(rt.local_cpu_slot(layer_id))
        except Exception:
            pass
        own = "-"
        if slot >= 0:
            if blk_lo is not None:
                is_own = blk_lo <= slot < blk_hi
            else:
                s = eplb_sets.get(layer_id)
                is_own = (slot in s) if s else None
            if is_own is None:
                own = "?"
            elif is_own:
                own = "yes"
                own_yes += 1
            else:
                own = "NO"
                own_no += 1
        py = _S.layer_py.get(layer_id, {})
        d = dlayers.get(layer_id) or dlayers.get(str(layer_id)) or {}
        calls = int(d.get("calls", 0))
        hit_calls = int(d.get("hit_calls", 0))
        toks = int(d.get("active_toks", 0))
        pairs = int(d.get("pairs", 0))
        gsum = int(d.get("gemm_ns", 0))
        gmax = int(d.get("gemm_max_ns", 0))
        avg_us = (gsum / 1000.0 / hit_calls) if hit_calls else 0.0
        tot_hit_toks += toks
        n_rows += 1
        mask = "OK" if py.get("armed", 0) > 0 else "NEVER"
        lines.append(
            "%5d %8d %5s %6s %8d %8d %9d %9d %10d %10d %8.1f %8.1f"
            % (
                layer_id,
                slot,
                own,
                mask,
                py.get("call", 0),
                py.get("armed", 0),
                calls,
                hit_calls,
                toks,
                pairs,
                avg_us,
                gmax / 1000.0,
            )
        )
    lines.append(_rule())
    if own_src == "block":
        lines.append(
            "layers=%d   cpu_slot is one of THIS rank's own GPU experts (block "
            "[%d, %d)) on %d, NOT on %d" % (n_rows, blk_lo, blk_hi, own_yes, own_no)
        )
    elif own_src == "eplb":
        lines.append(
            "layers=%d   cpu_slot is one of THIS rank's own GPU experts (EPLB "
            "slice, FALLBACK source) on %d, NOT on %d" % (n_rows, own_yes, own_no)
        )
    else:
        lines.append(
            "layers=%d   ownership not determinable -- no source was observed" % n_rows
        )
    if own_no:
        lines.append(
            "  -> on those %d layers the cpu_slot expert is owned by a DIFFERENT"
            % own_no
        )
        lines.append("     rank, which still computes it on GPU while this rank adds a")
        lines.append("     CPU result on top: every hit token there is counted twice.")
        lines.append("     The selector is supposed to choose from THIS rank's own")
        lines.append("     experts, so a non-zero count here means the ownership rule")
        lines.append(
            "     the selector used and the one this column used have diverged."
        )
        lines.append("     Compare the three sources below, and check the startup")
        lines.append("     assertion in model_load_hook.py. Reported, not repaired.")
    lines.append(
        "total CPU-side hit tokens (all layers, all replays) = %d" % tot_hit_toks
    )
    lines.append(_placement_disagreement(local_rank, module_any, eplb_sets))
    if not dlayers:
        if dstats.get("supported") is False:
            lines.append(
                "daemon columns are zero: the attached daemon predates slot v13"
            )
            lines.append(
                "  (rebuild and restart cpu_expertd to get per-layer numbers)."
            )
        elif dstats.get("trace_enabled") == 0:
            lines.append(
                "daemon columns are zero: trace_enabled was never set on the slot."
            )
        else:
            lines.append(
                "daemon columns are zero: no FORWARD_EXPERTS request was booked."
            )
    if dstats.get("overflow"):
        lines.append(
            "WARNING: %d requests had layer_id outside the traced range; rows "
            "are incomplete." % dstats["overflow"]
        )
    return "\n".join(lines)


def _placement_disagreement(
    local_rank: int, module_set: Optional[set], eplb_sets: dict
) -> str:
    """Compare the THREE answers to "which experts does this rank compute".

    They are not interchangeable, and only one of them decides anything:

      1. CONTIGUOUS BLOCK, rank r owns [r*n, (r+1)*n). Computed independently
         in two places that both matter --
         ``FusedMoE._map_global_expert_id_to_local_expert_id`` (which expert's
         weights land in which local slot) and ``StandardDispatcher.dispatch``
         (which builds ``local_expert_mapping``, sending everything outside
         the block to -1). This one is authoritative.
      2. ``FusedMoE._global_to_local_map``, built by DeepEPMoEHeto from its
         interleaved ``{eid: eid % ep_size}`` expert map. It is assigned and
         logged and NEVER READ -- there is no read site anywhere in
         ``python/``. Reasoning from it gives the opposite answer to (1).
      3. The EPLB ``physical_to_logical_map`` block slice, which is what
         ``fast_loop``'s candidate filter and ``cuda_graph_offload``'s
         local_expert_lo/hi actually read. It is derived from different
         metadata than (1), so agreement has to be checked, not assumed.
    """
    out: list[str] = []
    blk_r, blk_n = _S.block_rank, _S.block_num_local
    if blk_r is None or not blk_n:
        return (
            "gpu-expert ownership: the MoE module was never observed, so "
            "the authoritative block range is unknown."
        )
    lo, hi = blk_r * blk_n, (blk_r + 1) * blk_n
    blk_set = set(range(lo, hi))
    out.append("gpu-expert ownership cross-check (three sources, only one decides):")
    out.append(
        "  [1] BLOCK (authoritative): rank %d owns [%d, %d). Used by BOTH the"
        % (blk_r, lo, hi)
    )
    out.append("      weight loader and StandardDispatcher. The 'own?' column above")
    out.append("      follows this one.")

    # [2] the dead interleaved map
    if module_set is None:
        out.append("  [2] _global_to_local_map: not observed.")
    elif module_set == blk_set:
        out.append("  [2] _global_to_local_map: same set as [1] (and still unread).")
    else:
        only_mod = sorted(module_set - blk_set)[:6]
        out.append(
            "  [2] _global_to_local_map: DIFFERENT set (%d of %d ids differ,"
            % (len(module_set - blk_set), len(blk_set))
        )
        out.append("      e.g. %s). DeepEPMoEHeto builds it interleaved" % only_mod)
        out.append("      ({eid: eid % ep_size}), but NOTHING READS IT -- it is")
        out.append("      assigned and logged only. Anyone reasoning from this map")
        out.append("      (or from the log line it prints) reaches the OPPOSITE")
        out.append("      conclusion about which experts are local. Reported, not")
        out.append("      repaired.")

    # [3] the EPLB slice that fast_loop / cuda_graph_offload really read
    eplb_set = eplb_sets.get(0)
    if not eplb_set:
        out.append("  [3] EPLB block slice: metadata unavailable, cannot compare.")
    elif eplb_set == blk_set:
        out.append("  [3] EPLB block slice (read by fast_loop's candidate filter and")
        out.append("      cuda_graph_offload's local_expert_lo/hi): AGREES with [1]")
        out.append("      in THIS run -- but by coincidence, not by construction.")
        out.append("      [1] is moe_ep_rank * <module>.num_local_experts;")
        out.append("      fast_loop is rank * meta.num_local_physical_experts THEN")
        out.append("      mapped through physical_to_logical_map;")
        out.append("      cuda_graph_offload is the same product used as a RAW id")
        out.append("      range, without that mapping. The three coincide only while")
        out.append("      the map is identity, the two expert counts are equal, and")
        out.append("      moe_ep_rank == tp_rank. EPLB rebalancing, expert")
        out.append("      replication, or TP != EP would break any of those.")
    else:
        only_e = sorted(eplb_set - blk_set)[:6]
        only_b = sorted(blk_set - eplb_set)[:6]
        out.append(
            "  [3] EPLB block slice: DISAGREES with [1] -- has %s, misses %s."
            % (only_e, only_b)
        )
        out.append("      fast_loop and cuda_graph_offload read this one, so they are")
        out.append("      filtering/counting against the wrong range.")

    # Say what this section is NOT, so a clean result is not over-read.
    out.append("")
    out.append("  These three are a HAND-WRITTEN list of sources known to have")
    out.append("  disagreed before. Nothing here scans the tree: a fourth notion of")
    out.append("  'which experts are local' -- added later, or already present and")
    out.append("  never noticed -- would not appear, and this section would still")
    out.append("  print a clean result. [2] is in the list precisely because it looks")
    out.append("  authoritative and has zero read sites; that combination is found by")
    out.append("  grepping for read sites, not by running this report. Extend the")
    out.append("  list when you find the next one.")
    return "\n".join(out)


def _reselect_report() -> str:
    if not _S.reselects:
        return (
            "no re-selection event was recorded (the bootstrap assignment was\n"
            "never revised -- check CPU_OFFLOAD_SLOT_UPDATE_INTERVAL_MS and\n"
            "CPU_OFFLOAD_USE_LAST_GPU_SLOT)."
        )
    lines: list[str] = []
    lines.append("One line per event, layers sorted, with a digest of the change list.")
    lines.append(
        "`grep '^#'` gives one comparable line per event; `diff` on this block"
    )
    lines.append("localises where two runs parted company.")
    lines.append("Re-selection is gated on FORWARD PASSES, so two runs of the same")
    lines.append("config must produce identical lines here; a difference is a real")
    lines.append(
        "divergence, not timing jitter. (It used to be wall-clock gated, which"
    )
    lines.append("made every run differ.) CPU_OFFLOAD_SLOT_UPDATE_INTERVAL_MS=0 pins")
    lines.append("the layout entirely; CPU_OFFLOAD_SLOT_UPDATE_EVERY_N_FORWARDS sets")
    lines.append("the cadence.")
    lines.append(_rule())
    for i, ev in enumerate(_S.reselects):
        ch = ev["changes"]
        # A short digest of the change list. `grep '^#'` on two runs' reports
        # gives one comparable line per event, which localises a divergence
        # without reading every layer token.
        digest = hashlib.sha1(
            ",".join("%d:%d>%d" % c for c in ch).encode()
        ).hexdigest()[:10]
        lines.append(
            "#%-3d %-10s tick=%-5d fwd=%-6d changed=%-4d digest=%s src=%s"
            % (
                i,
                ev["event"],
                ev["tick"],
                ev["forward_no"],
                len(ch),
                digest,
                ev["source"] or "-",
            )
        )
        # Consecutive layers that made the SAME move collapse into one
        # token. The bootstrap typically moves all 94 layers identically, and
        # 94 identical tokens hide the one layer that differs.
        toks = []
        i = 0
        while i < len(ch):
            l0, o, n = ch[i]
            j = i
            while (
                j + 1 < len(ch)
                and ch[j + 1][1] == o
                and ch[j + 1][2] == n
                and ch[j + 1][0] == ch[j][0] + 1
            ):
                j += 1
            label = "L%d" % l0 if j == i else "L%d-%d" % (l0, ch[j][0])
            toks.append("%s:%d>%d" % (label, o, n))
            i = j + 1
        for j in range(0, len(toks), 6):
            lines.append("      " + " ".join("%-15s" % t for t in toks[j : j + 6]))
    lines.append(_rule())
    lines.append(
        "events=%d  total layer changes=%d"
        % (len(_S.reselects), sum(len(e["changes"]) for e in _S.reselects))
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def should_report(local_rank: int) -> bool:
    if not ENABLED:
        return False
    if _REPORT_RANKS is None:
        return True
    return int(local_rank) in _REPORT_RANKS


def emit_report(rt: Any) -> None:
    """Print the report once. Never raises."""
    if not ENABLED:
        return
    with _S.lock:
        if _S.reported:
            return
        _S.reported = True
    try:
        if not should_report(getattr(rt, "local_rank", -1)):
            return
        text = build_report(rt)
    except Exception:  # pragma: no cover - a broken report must not break exit
        text = "[moe_trace] report generation failed:\n" + traceback.format_exc()
    # print(), not logging: TP worker subprocess logging is not reliably
    # visible in this deployment, and the existing METRICS dump prints for the
    # same reason.
    print(text, flush=True)


def install_atexit(rt: Any) -> None:
    """Register the exit-time report. No-op when tracing is off."""
    if not ENABLED:
        return
    import atexit

    try:
        atexit.register(emit_report, rt)
    except Exception:
        pass


def reset_for_tests() -> None:
    """Drop all recorded state. Tests only."""
    global _S
    try:
        if _S.journal_fh is not None:
            _S.journal_fh.close()
    except Exception:
        pass
    _S = _State()
