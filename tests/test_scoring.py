#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评分口径的回归测试：专治"同义改写""日期格式""碎片词"。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refiner as R


def test_paraphrase_supported():
    """原文"延迟到18号"、节点"推迟到18日"——必须判为有支撑。"""
    raw = "交付日延迟到 18 号。"
    node = "推迟到 18 日。"
    s = R.support_ratio(node, raw)
    assert s >= 0.8, f"同义改写被判无支撑：{s}"
    print(f"PASS test_paraphrase_supported support={s:.2f}")


def test_date_formats_unified():
    a = R.normalize("2026-03-02 交付")
    b = R.normalize("3-02 交付")
    c = R.normalize("3月2日交付")
    assert a == b == c, (a, b, c)
    print("PASS test_date_formats_unified", a)


def test_fragments_not_extracted():
    """滑窗会抽出"供应商那"这类碎片；语料不够大时，抽词器直接闭嘴。"""
    text = "供应商那边说模具没到。供应商报价高。备选供应商可以按时交货。主供应商延后。" * 6
    terms = R.auto_terms(text, top=10)
    bad = [t for t in terms if t.endswith("那") or t.endswith("边") or t.endswith("的")]
    assert not bad, f"抽出了碎片：{bad} / 全部 {terms}"
    # 小语料（<3000 字）必须返回空，而不是硬凑
    assert R.auto_terms("供应商那边说模具没到。" * 2, top=10) == []
    print("PASS test_fragments_not_extracted", terms)


def test_entity_recall_ignores_noise():
    raw = "3月2日连续第4天游1000米，2026-03-05 交付200件。"
    node = "3月2日游了1000米；3月5日交付200件。"
    r, miss = R.entity_recall(raw, node)
    assert r >= 0.75, (r, miss)
    print(f"PASS test_entity_recall_ignores_noise recall={r:.2f}")


def test_score_is_script_only():
    raw = "3月2日谈定交付。3月5日改为3月18日。3月8日客户确认。" * 3
    node = "交付日从3月5日改到3月18日，客户已确认。"
    sc = R.score(raw, node)
    assert 0 <= sc["score"] <= 100
    assert sc["precision"] >= 0.5, sc
    print("PASS test_score_is_script_only", sc["score"], sc["recall"], sc["precision"])


if __name__ == "__main__":
    test_paraphrase_supported()
    test_date_formats_unified()
    test_fragments_not_extracted()
    test_entity_recall_ignores_noise()
    test_score_is_script_only()
    print("\n评分测试全部通过")
