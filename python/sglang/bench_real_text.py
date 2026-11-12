"""用真实文本拼出指定长度的基准输入。

MoE 把每个 token 交给哪几位专家，取决于 token 的内容。随机 token 测速度没问题，
拿来评估"挑热专家"却不可靠：随机输入下各专家的冷热分布和真实负载不一样，挑出来
的热专家也就不是真实场景里的那几位。所以长度按业界常用的组合定，内容从对话数据
集里取真实正文。

拼法：

1. 从数据集按顺序取非空正文（ShareGPT、ChatMed、JSONL、纯文本都认）。
2. 每个请求从上一个请求用完的下一条开始取，一条不够长就接着拼下一条，中间用空行
   隔开，同一批里的请求互不重复。
3. 套上对话模板，正文截到刚好让总长等于 ``input_len`` 个 token。

各组参数都从同一条记录开始取，所以 bs=1 的那一个请求和 bs=4 的第一个请求内容
相同，不同 batch 之间可以直接比。

模板头、正文、模板尾是分开分词再拼起来的，正文还在 token 边界上截断，所以拼出
的序列不一定和"把整段字符串交给分词器"的结果逐个相同。对测性能和看专家冷热
没有影响，但不要拿它去比对精度。

只依赖标准库，分词器由调用方传入（有 ``encode`` 的对象就行），所以能在没有 GPU、
没有模型的机器上单测。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence, Tuple, Union

BODY_PLACEHOLDER = "@@BENCH_REAL_TEXT_BODY@@"
RECORD_SEPARATOR = "\n\n"

_TURN_KEYS = ("conversations", "messages", "conversation")
_TEXT_KEYS = ("text", "prompt", "content", "question", "instruction", "input")


def parse_len_pairs(values: Iterable[str]) -> List[Tuple[int, int]]:
    """把 ``["1024:1024", "1024:8192,8192:1024"]`` 解析成去重后的长度组合。"""
    pairs: List[Tuple[int, int]] = []
    seen = set()
    for value in values:
        for item in re.split(r"[,\s]+", str(value).strip()):
            if not item:
                continue
            parts = item.split(":")
            try:
                if len(parts) != 2:
                    raise ValueError
                pair = (int(parts[0]), int(parts[1]))
            except ValueError:
                raise ValueError(
                    f"长度组合要写成 输入:输出，例如 1024:1024，收到 {item!r}"
                ) from None
            if pair[0] <= 0 or pair[1] <= 0:
                raise ValueError(f"输入和输出长度都必须是正数，收到 {item!r}")
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def segment_name(batch_size: int, input_len: int, output_len: int) -> str:
    return f"bs{int(batch_size)}-in{int(input_len)}-out{int(output_len)}"


def extract_text(record: Any) -> str:
    """从一条数据集记录里取出正文；取不到返回空串。"""
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return ""
    for key in _TURN_KEYS:
        turns = record.get(key)
        if not isinstance(turns, list):
            continue
        parts = []
        for turn in turns:
            if isinstance(turn, dict):
                value = turn.get("value", turn.get("content"))
            else:
                value = turn
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        if parts:
            return RECORD_SEPARATOR.join(parts)
    query = record.get("query")
    if isinstance(query, str) and query.strip():
        parts = [query.strip()]
        response = record.get("response")
        if isinstance(response, str) and response.strip():
            parts.append(response.strip())
        return RECORD_SEPARATOR.join(parts)
    for key in _TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_char(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as fh:
        while True:
            chunk = fh.read(4096)
            if not chunk:
                return ""
            stripped = chunk.lstrip()
            if stripped:
                return stripped[0]


def _iter_json_array(path: Path, chunk_size: int) -> Iterator[Any]:
    """逐条解析 JSON 数组，读够就停，不把几百 MB 的文件整个读进内存。"""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8-sig") as fh:
        buf = ""
        pos = 0
        eof = False

        def refill() -> bool:
            nonlocal buf, pos, eof
            more = fh.read(chunk_size)
            if not more:
                eof = True
                return False
            buf = buf[pos:] + more
            pos = 0
            return True

        while not eof and "[" not in buf:
            refill()
        if "[" not in buf:
            raise ValueError(f"{path} 不是 JSON 数组")
        pos = buf.index("[") + 1
        count = 0
        while True:
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf) or not refill():
                    break
            if pos >= len(buf):
                raise ValueError(f"{path} 的 JSON 数组没有结尾的 ]")
            if buf[pos] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError as exc:
                # 这一条被块边界截断了，读下一块再试；文件读完还解析不了就是坏数据。
                if refill():
                    continue
                # exc 里的行列号是相对缓冲区的，不是文件位置，所以报元素序号。
                raise ValueError(
                    f"{path} 的 JSON 数组第 {count + 1} 个元素不是合法的 JSON，"
                    f"文件可能没下载完整：{exc.msg}"
                ) from None
            if end == len(buf) and refill():
                # 数字这类值恰好在块尾时可能只读到一半，补一块重新解析。
                continue
            yield obj
            count += 1
            pos = end
            if pos > chunk_size:
                buf = buf[pos:]
                pos = 0


def _iter_jsonl(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_no} 行不是合法的 JSON：{exc}"
                ) from None


def _iter_paragraphs(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8-sig") as fh:
        lines: List[str] = []
        for line in fh:
            if line.strip():
                lines.append(line.rstrip("\n"))
            elif lines:
                yield "\n".join(lines)
                lines = []
        if lines:
            yield "\n".join(lines)


def iter_dataset_texts(
    path: Union[str, Path], chunk_size: int = 1 << 20
) -> Iterator[str]:
    """按文件顺序产出非空正文。

    ``.jsonl`` 或以 ``{`` 开头的文件按每行一条 JSON 读；以 ``[`` 开头的按 JSON
    数组流式读；其余按纯文本处理，空行分段。取不出正文的记录直接跳过，所以
    "第 N 条"指的是第 N 条非空正文，不是文件里的第 N 条记录。
    """
    path = Path(path)
    first = _first_char(path)
    if path.suffix == ".jsonl" or first == "{":
        records: Iterable[Any] = _iter_jsonl(path)
    elif first == "[":
        records = _iter_json_array(path, max(1, int(chunk_size)))
    else:
        records = _iter_paragraphs(path)
    for record in records:
        text = extract_text(record)
        if text:
            yield text


@dataclass
class BuiltBatch:
    input_ids: List[List[int]]
    # 每个请求用到的记录编号（第几条非空正文，从 0 数）。
    record_indices: List[List[int]]
    next_record: int


class RealTextPromptBuilder:
    """把真实正文拼成定长输入，分过词的记录会缓存，供后面各组参数复用。"""

    def __init__(
        self,
        tokenizer: Any,
        texts: Iterable[str],
        *,
        chat_template: bool = True,
        first_record_index: int = 0,
    ) -> None:
        self._tokenizer = tokenizer
        self._texts = iter(texts)
        self._record_tokens: List[List[int]] = []
        self.chat_template = bool(chat_template)
        self.first_record_index = int(first_record_index)
        self.prefix_ids, self.suffix_ids = self._template_parts()
        self.separator_ids = self._encode(RECORD_SEPARATOR)

    def _encode(self, text: str) -> List[int]:
        return [int(x) for x in self._tokenizer.encode(text, add_special_tokens=False)]

    def _template_parts(self) -> Tuple[List[int], List[int]]:
        if not self.chat_template:
            # 不套模板时只保留分词器自己要求的开头（例如 BOS），Qwen 没有。
            prefix = self._tokenizer.encode("", add_special_tokens=True)
            return [int(x) for x in prefix], []
        apply = getattr(self._tokenizer, "apply_chat_template", None)
        if apply is None:
            raise ValueError("分词器没有对话模板；不想套模板请加 --dataset-raw-text")
        try:
            rendered = apply(
                [{"role": "user", "content": BODY_PLACEHOLDER}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise ValueError(
                f"套对话模板失败：{exc}；不想套模板请加 --dataset-raw-text"
            ) from exc
        if not isinstance(rendered, str) or rendered.count(BODY_PLACEHOLDER) != 1:
            raise ValueError("对话模板渲染结果里找不到唯一的正文位置，没法按长度截断")
        head, tail = rendered.split(BODY_PLACEHOLDER)
        return self._encode(head), self._encode(tail)

    @property
    def template_len(self) -> int:
        return len(self.prefix_ids) + len(self.suffix_ids)

    def _record(self, index: int) -> List[int]:
        while len(self._record_tokens) <= index:
            try:
                text = next(self._texts)
            except StopIteration:
                raise ValueError(
                    f"数据集从第 {self.first_record_index} 条起只有 "
                    f"{len(self._record_tokens)} 条可用正文，不够拼出这一批输入"
                ) from None
            self._record_tokens.append(self._encode(text))
        return self._record_tokens[index]

    def build_batch(
        self, batch_size: int, input_len: int, start: int = 0
    ) -> BuiltBatch:
        """拼 ``batch_size`` 个长度正好是 ``input_len`` 的输入。

        ``start`` 是相对 ``first_record_index`` 的位置；返回的记录编号已经加上
        ``first_record_index``，可以直接拿去数据集里找原文。
        """
        body_len = int(input_len) - self.template_len
        if body_len <= 0:
            raise ValueError(
                f"输入长度 {input_len} 放不下对话模板本身（模板占 {self.template_len} 个 token）"
            )
        cursor = int(start)
        input_ids: List[List[int]] = []
        record_indices: List[List[int]] = []
        for _ in range(int(batch_size)):
            body: List[int] = []
            used: List[int] = []
            while len(body) < body_len:
                if used:
                    body.extend(self.separator_ids)
                    if len(body) >= body_len:
                        break
                body.extend(self._record(cursor))
                used.append(self.first_record_index + cursor)
                cursor += 1
            input_ids.append(self.prefix_ids + body[:body_len] + self.suffix_ids)
            record_indices.append(used)
        return BuiltBatch(input_ids, record_indices, cursor)


def group_by_token_budget(lengths: Sequence[int], budget: int) -> List[List[int]]:
    """按整个请求分组，每组 token 总数不超过 ``budget``。

    单个请求本身就超过上限时独占一组；``budget`` 不大于 0 表示不分组。
    """
    lengths = [int(x) for x in lengths]
    if not lengths:
        return []
    if budget <= 0:
        return [list(range(len(lengths)))]
    groups: List[List[int]] = []
    current: List[int] = []
    used = 0
    for index, length in enumerate(lengths):
        if current and used + length > budget:
            groups.append(current)
            current, used = [], 0
        current.append(index)
        used += length
    groups.append(current)
    return groups


def trace_sample_every(output_len: int, target_steps: int) -> int:
    """让一段输出大约记下 ``target_steps`` 个 decode 步时的抽样间隔。

    第一个输出 token 由预填充产生，所以 decode 一共 ``output_len - 1`` 步。
    ``target_steps`` 不大于 0 表示每步都记。
    """
    decode_steps = max(1, int(output_len) - 1)
    if target_steps <= 0:
        return 1
    return max(1, math.ceil(decode_steps / int(target_steps)))


def input_digest(input_ids: Sequence[Sequence[int]]) -> str:
    """整批输入的 sha256，用来确认各 rank 拿到的输入一模一样。"""
    payload = json.dumps(
        [[int(x) for x in ids] for ids in input_ids], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
