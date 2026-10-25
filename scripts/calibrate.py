#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校准：用"已知缺陷梯度"验证 score 与真实质量是否同向。

做法：拿同一个原文，造 6 个质量递减的节点（不丢 / 丢 25% / 丢 50% / 丢 75% /
全编 / 照抄原文），算分，再看分与"质量等级"的秩相关。

这不是人工标注，但它是**可以复现的客观梯度**；人工标注见 README 的 Roadmap。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refiner as R

RAW = Path(__file__).resolve().parents[1] / "examples" / "raw_2026-03_delivery.md"
NODE = Path(__file__).resolve().parents[1] / "examples" / "node_example.json"


def variants(raw: str) -> list:
    base = json.loads(NODE.read_text(encoding="utf-8"))
    facts = base["facts"]
    out = []
    for name, keep in (("完整", 1.0), ("丢25%", 0.75), ("丢50%", 0.5), ("丢75%", 0.25)):
        n = int(len(facts) * keep)
        out.append((name, {"facts": facts[:n], "persona_anchors": base["persona_anchors"]}))
    out.append(("全编", {"facts": [{"text": "一切都很顺利，没有问题，客户很满意。"}], "persona_anchors": []}))
    out.append(("照抄", {"__raw__": True}))
    return out


def rank(xs: list) -> list:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


def spearman(a: list, b: list) -> float:
    ra, rb = rank(a), rank(b)
    n = len(a)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main() -> int:
    raw = RAW.read_text(encoding="utf-8")
    rows = []
    for i, (name, node) in enumerate(variants(raw)):
        if node.get("__raw__"):
            text = raw
        else:
            text = R.node_to_text(node)
        sc = R.score(raw, text, claim_text=None if node.get("__raw__") else R.node_claims(node))
        rows.append((name, sc["score"], sc["recall"], sc["precision"], sc["compression_ratio"]))
        print(f"{name:<6} 分数 {sc['score']:>5} ｜ 召回 {sc['recall']:.2f} "
              f"｜ 精确 {sc['precision']:.2f} ｜ 压缩率 {sc['compression_ratio']:.2f}")

    quality = [5, 4, 3, 2, 1, 0]          # 构造的"真实质量"梯度：完整最好，照抄最差
    scores = [r[1] for r in rows]
    rho = spearman(quality, scores)
    print(f"\nSpearman ρ（质量等级 vs 分数）= {rho:.3f}")
    if rho >= 0.8:
        print("=> 分数与质量强同向，可以作为横向比较依据。")
    elif rho >= 0.5:
        print("=> 中等相关，能排序但不能当绝对分。")
    else:
        print("=> 相关不足，评分公式还要再改。")
    return 0 if rho >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
