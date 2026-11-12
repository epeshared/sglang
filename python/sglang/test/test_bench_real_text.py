"""bench_real_text 的单测：只用标准库和假分词器，不需要模型或 GPU。"""

from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path

from sglang import bench_real_text as brt


class _CharTokenizer:
    """一个字符一个 token；带特殊 token 时在开头补 1，模拟 BOS。"""

    def __init__(self, template=lambda body: f"<u>{body}</u><a>"):
        self._template = template

    def encode(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        return ([1] + ids) if add_special_tokens else ids

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        assert tokenize is False and add_generation_prompt is True
        return self._template(messages[0]["content"])


def _decode(ids):
    return "".join(chr(x) for x in ids)


class ParseAndExtractTests(unittest.TestCase):
    def test_parse_len_pairs_dedups_and_keeps_order(self):
        self.assertEqual(
            brt.parse_len_pairs(["1024:1024", "1024:8192,8192:1024", " 1024:1024 "]),
            [(1024, 1024), (1024, 8192), (8192, 1024)],
        )
        self.assertEqual(brt.parse_len_pairs(["8:4 16:2"]), [(8, 4), (16, 2)])
        for bad in ["1024", "a:b", "0:5", "5:-1", "1:2:3"]:
            with self.assertRaises(ValueError, msg=bad):
                brt.parse_len_pairs([bad])

    def test_segment_name(self):
        self.assertEqual(brt.segment_name(4, 1024, 8192), "bs4-in1024-out8192")

    def test_extract_text_formats(self):
        sharegpt = {
            "id": "x",
            "conversations": [
                {"from": "human", "value": " 你好 "},
                {"from": "gpt", "value": "hello"},
                {"from": "gpt", "value": "   "},
            ],
        }
        self.assertEqual(brt.extract_text(sharegpt), "你好\n\nhello")
        messages = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        self.assertEqual(brt.extract_text(messages), "q\n\na")
        self.assertEqual(
            brt.extract_text({"query": "头疼", "response": "多休息"}), "头疼\n\n多休息"
        )
        self.assertEqual(brt.extract_text({"query": "头疼"}), "头疼")
        self.assertEqual(brt.extract_text({"text": " body "}), "body")
        self.assertEqual(brt.extract_text({"instruction": "do it"}), "do it")
        self.assertEqual(brt.extract_text(" plain "), "plain")
        self.assertEqual(brt.extract_text({"conversations": []}), "")
        self.assertEqual(brt.extract_text({"unknown": "x"}), "")
        self.assertEqual(brt.extract_text(42), "")


class DatasetReadingTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_json_array_streams_across_tiny_chunks(self):
        records = [
            {
                "conversations": [
                    {"from": "human", "value": '第一条 "引号" 和 \\ 反斜杠'}
                ]
            },
            {"text": "x" * 50, "n": 1234567890, "f": -1.5e3, "ok": True, "z": None},
            {"query": "q", "response": "r", "nested": {"a": [1, 2, {"b": "]"}]}},
            {"unknown": "跳过"},
            "  纯字符串记录  ",
        ]
        path = self.dir / "data.json"
        path.write_text(
            "\ufeff \n" + json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        expected = [t for t in (brt.extract_text(r) for r in records) if t]
        for chunk_size in [1, 7, 64, 1 << 20]:
            self.assertEqual(
                list(brt.iter_dataset_texts(path, chunk_size=chunk_size)), expected
            )

    def test_json_array_stops_early_without_reading_bad_tail(self):
        path = self.dir / "data.json"
        path.write_text(
            '[{"text": "a"}, {"text": "b"}, {"text": "broken', encoding="utf-8"
        )
        self.assertEqual(
            list(itertools.islice(brt.iter_dataset_texts(path, chunk_size=4), 2)),
            ["a", "b"],
        )
        for chunk_size in [4, 1 << 20]:
            with self.assertRaisesRegex(ValueError, "第 3 个元素.*没下载完整"):
                list(brt.iter_dataset_texts(path, chunk_size=chunk_size))

    def test_json_array_without_closing_bracket_is_an_error(self):
        path = self.dir / "data.json"
        path.write_text('[{"text": "a"}, ', encoding="utf-8")
        with self.assertRaises(ValueError):
            list(brt.iter_dataset_texts(path))

    def test_jsonl_by_suffix_or_first_char(self):
        lines = ['{"text": "a"}', "", '{"text": ""}', '{"query": "b"}']
        for name in ["data.jsonl", "data.json"]:
            path = self.dir / name
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(list(brt.iter_dataset_texts(path)), ["a", "b"])
        bad = self.dir / "bad.jsonl"
        bad.write_text('{"text": "a"}\nnot json\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            list(brt.iter_dataset_texts(bad))

    def test_plain_text_splits_on_blank_lines(self):
        path = self.dir / "book.txt"
        path.write_text("第一段\n续行\n\n\n第二段\n   \n第三段", encoding="utf-8")
        self.assertEqual(
            list(brt.iter_dataset_texts(path)), ["第一段\n续行", "第二段", "第三段"]
        )


class PromptBuilderTests(unittest.TestCase):
    def test_exact_length_and_distinct_consecutive_records(self):
        texts = ["aaaa", "bbbbbbbbbbbbbbb", "cc", "ddd", "eeeeeeeeeeee", "ffffffffffff"]
        builder = brt.RealTextPromptBuilder(
            _CharTokenizer(), texts, first_record_index=100
        )
        self.assertEqual(_decode(builder.prefix_ids), "<u>")
        self.assertEqual(_decode(builder.suffix_ids), "</u><a>")
        batch = builder.build_batch(3, 20)

        # 模板占 10 个 token，正文 10 个。
        self.assertTrue(all(len(ids) == 20 for ids in batch.input_ids))
        bodies = [_decode(ids)[3:-7] for ids in batch.input_ids]
        self.assertEqual(bodies[0], "aaaa\n\nbbbb")
        self.assertEqual(bodies[1], "cc\n\nddd\n\ne")
        self.assertEqual(bodies[2], "ffffffffff")
        self.assertEqual(batch.record_indices, [[100, 101], [102, 103, 104], [105]])
        self.assertEqual(batch.next_record, 6)

    def test_separator_filling_the_body_does_not_claim_a_record(self):
        builder = brt.RealTextPromptBuilder(
            _CharTokenizer(), ["abcd", "zzz"], chat_template=False
        )
        batch = builder.build_batch(1, 1 + 5)
        self.assertEqual(_decode(batch.input_ids[0][1:]), "abcd\n")
        self.assertEqual(batch.record_indices, [[0]])
        self.assertEqual(batch.next_record, 1)

    def test_same_start_gives_same_prompts_and_tokenizes_once(self):
        pulled = []

        def texts():
            for i in range(10):
                pulled.append(i)
                yield f"record-{i}-" + "x" * 8

        builder = brt.RealTextPromptBuilder(_CharTokenizer(), texts())
        small = builder.build_batch(1, 30)
        large = builder.build_batch(4, 30)
        self.assertEqual(small.input_ids[0], large.input_ids[0])
        self.assertEqual(pulled, list(range(len(pulled))))
        self.assertEqual(len(pulled), large.next_record)

    def test_raw_text_mode_keeps_only_bos(self):
        builder = brt.RealTextPromptBuilder(
            _CharTokenizer(), ["hello world"], chat_template=False
        )
        self.assertEqual(builder.prefix_ids, [1])
        self.assertEqual(builder.suffix_ids, [])
        self.assertEqual(
            builder.build_batch(1, 6).input_ids, [[1] + [ord(c) for c in "hello"]]
        )

    def test_errors_are_explicit(self):
        builder = brt.RealTextPromptBuilder(_CharTokenizer(), ["abc"])
        with self.assertRaises(ValueError):
            builder.build_batch(1, 10)
        with self.assertRaises(ValueError):
            builder.build_batch(2, 14)
        with self.assertRaises(ValueError):
            brt.RealTextPromptBuilder(
                _CharTokenizer(template=lambda body: "no body"), []
            )
        with self.assertRaises(ValueError):
            brt.RealTextPromptBuilder(
                _CharTokenizer(template=lambda body: body + body), []
            )

        class NoTemplate:
            def encode(self, text, add_special_tokens=True):
                return [ord(c) for c in text]

        with self.assertRaises(ValueError):
            brt.RealTextPromptBuilder(NoTemplate(), [])
        self.assertEqual(
            brt.RealTextPromptBuilder(NoTemplate(), ["xyz"], chat_template=False)
            .build_batch(1, 2)
            .input_ids,
            [[ord("x"), ord("y")]],
        )


class HelperTests(unittest.TestCase):
    def test_group_by_token_budget(self):
        self.assertEqual(brt.group_by_token_budget([8192] * 3, 8192), [[0], [1], [2]])
        self.assertEqual(brt.group_by_token_budget([1024] * 8, 8192), [list(range(8))])
        self.assertEqual(
            brt.group_by_token_budget([1024] * 9, 8192), [list(range(8)), [8]]
        )
        self.assertEqual(
            brt.group_by_token_budget([10000, 10, 10], 8192), [[0], [1, 2]]
        )
        self.assertEqual(brt.group_by_token_budget([5, 5], 0), [[0, 1]])
        self.assertEqual(brt.group_by_token_budget([], 8192), [])

    def test_trace_sample_every(self):
        self.assertEqual(brt.trace_sample_every(1024, 64), 16)
        self.assertEqual(brt.trace_sample_every(8192, 64), 128)
        self.assertEqual(brt.trace_sample_every(65, 64), 1)
        self.assertEqual(brt.trace_sample_every(66, 64), 2)
        self.assertEqual(brt.trace_sample_every(1024, 0), 1)
        self.assertEqual(brt.trace_sample_every(1, 64), 1)

    def test_input_digest(self):
        digest = brt.input_digest([[1, 2], [3]])
        self.assertEqual(digest, brt.input_digest([[1, 2], [3]]))
        self.assertNotEqual(digest, brt.input_digest([[1, 2, 3]]))
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
