#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""健壮性测试：token 水位、滑动窗口、烂 JSON 修复。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refiner as R


def test_token_count_monotonic():
    a = R.count_tokens("短文本")
    b = R.count_tokens("短文本" * 100)
    assert b > a > 0, (a, b)
    print(f"PASS test_token_count_monotonic {a} -> {b}")


def test_watermark_triggers():
    big = "这是一段测试文本，用来把上下文顶到水位以上。" * 9000
    rep = R.token_report(big)
    assert rep["over_watermark"], rep
    print(f"PASS test_watermark_triggers tokens={rep['tokens']} ratio={rep['ratio']}")


def test_split_window_keeps_tail():
    text = "\n\n".join([f"第{i}轮：{'内容' * 20}" for i in range(1, 11)])
    older, recent = R.split_window(text, keep_turns=3)
    assert "第10轮" in recent and "第1轮" not in recent, recent[:80]
    assert "第1轮" in older
    print("PASS test_split_window_keeps_tail", len(older), len(recent))


def test_repair_broken_json():
    bad = '```json\n{ title: "x", "facts": [{"text": "a",},], "ok": True, }\n```'
    fixed = R.repair_json(bad)
    got = R.extract_json(bad)
    assert got["title"] == "x" and got["ok"] is True, got
    assert '"title"' in fixed
    print("PASS test_repair_broken_json", fixed[:60])


def test_extract_json_with_prose():
    s = '好的，这是结果：\n{"a": 1, "b": [1,2,]}\n希望有帮助！'
    assert R.extract_json(s) == {"a": 1, "b": [1, 2]}
    print("PASS test_extract_json_with_prose")


if __name__ == "__main__":
    test_token_count_monotonic()
    test_watermark_triggers()
    test_split_window_keeps_tail()
    test_repair_broken_json()
    test_extract_json_with_prose()
    print("\n健壮性测试全部通过")
