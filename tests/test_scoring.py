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


def test_real_words_extracted():
    """正向用例：高频三字词必须抽得出来（这条曾经是红的）。"""
    text = "甲方与供应商约定，供应商需要在三月二日交付第一批货。" * 120
    terms = R.auto_terms(text, top=12)
    assert "供应商" in terms, f"高频三字词没抽出来：{terms}"
    print("PASS test_real_words_extracted", terms[:6])


def test_sentence_initial_not_killed():
    """句首出现、没有左邻的词，不该因为左熵为 0 被判死。"""
    text = "供应商报价高。供应商发货慢。供应商说可以。供应商又改口。供应商最后同意了。" * 150
    terms = R.auto_terms(text, top=10)
    assert "供应商" in terms, f"句首词被误杀：{terms}"
    print("PASS test_sentence_initial_not_killed", terms[:5])


def test_antonyms_not_merged():
    """降价 与 提价 是反义，归一化不许把它们合并。"""
    assert R.normalize("降价") != R.normalize("提价"), "反义词被合并了"
    assert R.normalize("涨价") == R.normalize("提价"), "同义词应合并"
    print("PASS test_antonyms_not_merged", R.normalize("降价"), R.normalize("提价"))


def test_fabricated_clause_unsupported():
    """节点里凭空多出来的日期，必须被判无支撑。"""
    raw = "3月2日谈定交付。3月5日改期。3月8日客户确认。"
    assert R.support_ratio("从3月5日改到3月18日", raw) < 0.5, "编造的日期被判成有支撑"
    print("PASS test_fabricated_clause_unsupported")


if __name__ == "__main__":
    test_paraphrase_supported()
    test_date_formats_unified()
    test_fragments_not_extracted()
    test_entity_recall_ignores_noise()
    test_score_is_script_only()
    test_real_words_extracted()
    test_sentence_initial_not_killed()
    test_antonyms_not_merged()
    test_fabricated_clause_unsupported()
    print("\n评分测试全部通过")
