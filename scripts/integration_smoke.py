#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""联调测试：真的调模型，验证 compress / probe / judge 三条链路。

离线测试证明逻辑对；这个脚本证明"模型真的返回了能解析的东西"。
需要 DEEPSEEK_API_KEY（或本机 DSH 凭证）。judge 需要智谱 key，限流时记为 SKIP。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refiner as R

DOC = Path(__file__).resolve().parents[1] / "examples" / "raw_2026-03_delivery.md"


def main() -> int:
    raw = DOC.read_text(encoding="utf-8")
    key = R.deepseek_key()
    if not key:
        print("SKIP：没有 DeepSeek key")
        return 0

    print("① compress …")
    try:
        node, chunks = R.compress_pool(raw, key)
    except Exception as e:
        print(f"FAIL compress：{type(e).__name__}: {e}")
        return 1
    facts = node.get("facts", [])
    assert isinstance(facts, list) and facts, "没抽到事实"
    assert all(isinstance(f, dict) and f.get("text") for f in facts), "事实结构不对"
    print(f"   OK：{len(facts)} 条事实，{chunks} 块，标题「{node.get('node_title','')}」")

    text = R.node_to_text(node)
    sc = R.score(raw, text, claim_text=R.node_claims(node))
    print(f"   分数 {sc['score']}（召回 {sc['recall']} / 精确 {sc['precision']}）")

    print("② probe …")
    pv = R.probe(raw, text, key, k=3)
    if pv.get("probe") is None:
        print(f"   SKIP：{pv.get('error')}")
    else:
        print(f"   OK：探针 {pv['probe']}（{sum(1 for x in pv['qa'] if x['hit'])}/{len(pv['qa'])} 命中）")
        for qa in pv["qa"]:
            mark = "✅" if qa["hit"] else "❌"
            print(f"   {mark} {qa['q']} → 答「{qa['a']}」｜标准「{qa['gold']}」")

    print("③ judge …")
    gk = R.glm_key()
    if not gk:
        print("   SKIP：没有智谱 key")
    else:
        try:
            out = R.chat(R.GLM_URL, gk, R.GLM_MODEL, R.JUDGE_SYS,
                         f"【原文】\n{raw}\n\n【节点】\n{text}\n\n输出 JSON。", max_tokens=600)
            v = R.extract_json(out)
            print(f"   OK：漏 {len(v.get('missed_facts', []))} 条 / 无支撑 "
                  f"{len(v.get('unsupported_claims', []))} 条 / 矛盾 {len(v.get('contradictions', []))} 条")
        except Exception as e:
            print(f"   SKIP：{type(e).__name__}（多为免费档限流）")

    print("\n联调完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
