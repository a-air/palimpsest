#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事实版本链的最小回归测试（不走网络，不碰真实归档）。"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refiner as R


def fresh_store():
    tmp = Path(tempfile.mkdtemp(prefix="refiner-test-"))
    R.FACTS = tmp / "facts.jsonl"
    return tmp


def test_add_then_supersede():
    fresh_store()
    old = [{"text": "项目主线在 3 月 12 日交付，第一批 200 件。", "kind": "fact", "confidence": "S"}]
    st1 = R.merge_facts(old, "archive/2026-03-02", key="", now_iso="2026-03-02T10:00:00+08:00")
    assert st1["added"] == 1, st1

    new = [{"text": "项目已改为 3 月 18 日交付，第一批调整为 300 件。", "kind": "fact", "confidence": "A"}]
    st2 = R.merge_facts(new, "archive/2026-03-08", key="", now_iso="2026-03-08T10:00:00+08:00")
    rows = R.load_facts()
    assert len([r for r in rows if r["status"] == "active"]) == 1, rows
    assert len([r for r in rows if r["status"] == "superseded"]) == 1, rows
    old_row = [r for r in rows if r["status"] == "superseded"][0]
    assert old_row["valid_to"] == "2026-03-08T10:00:00+08:00"
    assert old_row["superseded_by"], "缺少 superseded_by"
    print("PASS test_add_then_supersede", st2)


def test_duplicate_not_added():
    fresh_store()
    f = [{"text": "样品到货时间推迟到 3 月 12 日。", "kind": "fact", "confidence": "S"}]
    R.merge_facts(f, "archive/a", key="", now_iso="2026-03-02T00:00:00+08:00")
    st = R.merge_facts(f, "archive/b", key="", now_iso="2026-03-03T00:00:00+08:00")
    rows = R.load_facts()
    assert st["duplicate"] == 1, st
    assert len(rows) == 1, rows
    print("PASS test_duplicate_not_added", st)


def test_unrelated_added():
    fresh_store()
    R.merge_facts([{"text": "供应商 A 的模具只够做 300 件。"}],
                  "archive/a", key="", now_iso="2026-03-02T00:00:00+08:00")
    st = R.merge_facts([{"text": "质检流程定在下周一上午。"}],
                       "archive/b", key="", now_iso="2026-03-03T00:00:00+08:00")
    assert st["added"] == 1 and st["superseded"] == 0, st
    assert len(R.load_facts()) == 2
    print("PASS test_unrelated_added", st)


def test_history_chain():
    fresh_store()
    a = "交付地址在旧仓库。"
    b = "交付地址已改成新仓库，走东门。"
    R.merge_facts([{"text": a}], "archive/a", key="", now_iso="2026-03-02T00:00:00+08:00")
    R.merge_facts([{"text": b}], "archive/b", key="", now_iso="2026-03-08T00:00:00+08:00")
    rows = R.load_facts()
    old = [r for r in rows if r["status"] == "superseded"]
    assert old, "应当有一条被覆盖"
    assert old[0]["superseded_by"], "链断了"
    print("PASS test_history_chain")


def test_no_false_supersede():
    """同一个项目的两件不同的事，不许互相覆盖。"""
    fresh_store()
    R.merge_facts([{"text": "供应商 A 报价比原来高 8%。"}], "archive/a", key="",
                  now_iso="2026-03-02T00:00:00+08:00")
    st = R.merge_facts([{"text": "客户回复可以接受新方案。"}], "archive/b", key="",
                       now_iso="2026-03-03T00:00:00+08:00")
    rows = R.load_facts()
    assert st["superseded"] == 0, st
    assert len([r for r in rows if r["status"] == "active"]) == 2, rows
    print("PASS test_no_false_supersede", st)


if __name__ == "__main__":
    test_add_then_supersede()
    test_duplicate_not_added()
    test_unrelated_added()
    test_history_chain()
    test_no_false_supersede()
    print("\n全部通过")
