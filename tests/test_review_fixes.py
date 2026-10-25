#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""针对代码评审六条意见的回归测试（每一条都对应一个断言）。"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refiner as R

RAW = Path(__file__).resolve().parents[1] / "examples" / "raw_2026-03_delivery.md"
NODE = Path(__file__).resolve().parents[1] / "examples" / "node_example.json"


def test_empty_node_scores_zero():
    """① 空节点不许白拿分。"""
    raw = RAW.read_text(encoding="utf-8")
    sc = R.score(raw, "")
    assert sc["score"] <= 5, sc
    print("PASS test_empty_node_scores_zero", sc["score"])


def test_copy_beats_nothing_but_loses_to_summary():
    """① 照抄原文应明显低于认真压缩的节点。"""
    raw = RAW.read_text(encoding="utf-8")
    node = R.node_to_text(json.loads(NODE.read_text(encoding="utf-8")))
    copy = R.score(raw, raw)["score"]
    good = R.score(raw, node)["score"]
    assert copy < good, f"照抄 {copy} 不该高于认真压缩 {good}"
    assert 40 <= copy <= 60, f"照抄应落在 50 附近，实际 {copy}"
    print(f"PASS test_copy_beats_nothing_but_loses_to_summary copy={copy} good={good}")


def test_probe_rejects_stale_value():
    """② 拿旧日期作答必须判错，哪怕这句话原文里确实出现过。"""
    assert R.answer_matches("3月12日", "3月18日") is False
    assert R.answer_matches("3月18日", "3月18日") is True
    assert R.answer_matches("200件", "300件") is False
    assert R.answer_matches("节点未记录", "3月18日") is False
    print("PASS test_probe_rejects_stale_value")


def test_chunking_never_truncates():
    """③ 长文本必须切块喂模型，不许静默截断。"""
    text = "\n\n".join(f"第{i}段：" + "内容" * 300 for i in range(20))
    chunks = R.chunk_text(text, 4000)
    assert len(chunks) > 1
    assert sum(len(c) for c in chunks) >= len(text) * 0.95, "切块丢了内容"
    assert "第19段" in "".join(chunks), "尾部被截断了"
    print(f"PASS test_chunking_never_truncates chunks={len(chunks)}")


def test_save_facts_is_atomic():
    """④ 写事实库必须是临时文件 + replace。"""
    tmp = Path(tempfile.mkdtemp(prefix="facts-"))
    old = R.FACTS
    R.FACTS = tmp / "facts.jsonl"
    try:
        rows = [{"id": "a", "text": "x", "status": "active"}]
        R.save_facts(rows)
        assert json.loads(R.FACTS.read_text(encoding="utf-8").strip())["id"] == "a"
        assert not list(tmp.glob("*.tmp")), "临时文件没清掉"
        R.save_facts(rows + [{"id": "b", "text": "y", "status": "active"}])
        assert len(R.FACTS.read_text(encoding="utf-8").strip().splitlines()) == 2
    finally:
        R.FACTS = old
    print("PASS test_save_facts_is_atomic")


def test_extract_json_not_greedy():
    """⑤ JSON 后面再跟一段带大括号的话，不许解析失败。"""
    s = '结果如下：{"a": 1}\n补充说明：{这不是 JSON}'
    assert R.extract_json(s)["a"] == 1
    print("PASS test_extract_json_not_greedy")


def test_no_local_paths_in_source():
    """⑥ 源码里不许出现本机路径/发行版/用户名（这里也不直接写它们）。"""
    src = (Path(__file__).resolve().parents[1] / "refiner.py").read_text(encoding="utf-8")
    forbidden = ["/ho" + "me/", "Ubuntu-" + "24.04", "fu" + "guang", "C:\\Us" + "ers"]
    for bad in forbidden:
        assert bad not in src, "源码里还留着本机路径片段"
    print("PASS test_no_local_paths_in_source")


if __name__ == "__main__":
    test_empty_node_scores_zero()
    test_copy_beats_nothing_but_loses_to_summary()
    test_probe_rejects_stale_value()
    test_chunking_never_truncates()
    test_save_facts_is_atomic()
    test_extract_json_not_greedy()
    test_no_local_paths_in_source()
    print("\n评审意见回归测试全部通过")
