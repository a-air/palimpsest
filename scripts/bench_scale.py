#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实规模验证：造 N 个批次，测 index / search / 抽词 的耗时。

用法：python scripts/bench_scale.py [N]
默认 N=3000（README 里立的目标规模）。
"""
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refiner as R

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
random.seed(7)

WORDS = ("供应商 交付 样品 焊接 质检 客户 模具 排期 返工 合同 物流 报关 对账 库存 批次").split()


def build(root: Path, n: int):
    (root / "archive").mkdir()
    (root / "cold").mkdir()
    (root / "state").mkdir()
    day = root / "archive" / "2026-06-01"
    day.mkdir()
    for i in range(n):
        facts = []
        for k in range(8):
            w = random.choice(WORDS)
            facts.append({"text": f"{w}第{i}批次记录：{w}进度{random.randint(1,100)}%，"
                                  f"责任人{random.choice(WORDS)}，"
                                  f"3月{random.randint(1,28)}日更新。",
                          "kind": "fact", "confidence": random.choice("SAB")})
        node = {"node_title": f"批次{i}", "summary": " ".join(random.sample(WORDS, 4)),
                "facts": facts, "persona_anchors": [], "open_threads": []}
        (day / f"{i:05d}_node.json").write_text(json.dumps(node, ensure_ascii=False), encoding="utf-8")
        (day / f"{i:05d}_meta.json").write_text(json.dumps(
            {"at": "2026-06-01T10:00:00+08:00", "score": {"score": random.randint(40, 90)}},
            ensure_ascii=False), encoding="utf-8")


def main():
    root = Path(tempfile.mkdtemp(prefix="bench-"))
    R.ROOT, R.ARCHIVE, R.COLD, R.STATE = root, root / "archive", root / "cold", root / "state"
    R.DB = root / "state" / "memory.db"
    R.FACTS = root / "state" / "facts.jsonl"

    print(f"生成 {N} 个批次…")
    t0 = time.time()
    build(root, N)
    print(f"  生成耗时 {time.time()-t0:.1f}s")

    t0 = time.time()
    R.cmd_index(type("A", (), {"cold": False})())
    idx = time.time() - t0

    conn = R.db()
    total = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    size = R.DB.stat().st_size / 1024 / 1024
    conn.close()

    queries = ["供应商", "交付日", "质检", "批次42", "报关"]
    times = []
    for q in queries * 4:
        conn = R.db()
        t0 = time.time()
        if len(q) >= 3:
            conn.execute("SELECT path FROM nodes_fts WHERE nodes_fts MATCH ? LIMIT 10", (f'"{q}"',)).fetchall()
        else:
            conn.execute("SELECT path FROM nodes WHERE text LIKE ? LIMIT 10", (f"%{q}%",)).fetchall()
        times.append((time.time() - t0) * 1000)
        conn.close()

    # 抽词性能：20 万字语料
    big = "".join(random.choice(WORDS) + "第" + str(random.randint(1, 999)) + "批次，" for _ in range(20000))
    t0 = time.time()
    terms = R.auto_terms(big, top=10)
    lex = time.time() - t0

    print("\n=== 结果 ===")
    print(f"节点数        {total}")
    print(f"库大小        {size:.1f} MB")
    print(f"index 总耗时  {idx:.2f}s（{total/idx:.0f} 节点/秒）")
    print(f"search 平均   {sum(times)/len(times):.1f} ms ｜ 最慢 {max(times):.1f} ms")
    print(f"抽词 20万字   {lex:.2f}s ｜ 抽出 {len(terms)} 个词：{terms[:5]}")
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
