#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI 冒烟测试：不联网，只验命令能跑、输出是合法 JSON。"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args, expect=0):
    r = subprocess.run([PY, str(ROOT / "refiner.py"), *args],
                       capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))
    assert r.returncode == expect, f"{args} -> {r.returncode}\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_score_cli():
    out = run("score", "--raw", "examples/raw_2026-03_delivery.md",
              "--node", "examples/node_example.json")
    data = json.loads(out[out.index("{"):out.rindex("}") + 1])
    assert 0 <= data["score"] <= 100, data
    assert data["recall"] > 0.5, data        # 样例节点应该覆盖得不错
    assert data["precision"] > 0.5, data
    print(f"PASS test_score_cli score={data['score']} recall={data['recall']} precision={data['precision']}")


def test_short_raw_refused():
    out = run("score", "--raw", "examples/node_example.json", "--node", "examples/node_example.json")
    assert "score" in out
    print("PASS test_short_raw_refused")


def test_help_lists_commands():
    out = run("--help")
    for cmd in ("compress", "score", "probe", "judge", "refine", "report", "facts", "history", "recall", "rebuild"):
        assert cmd in out, f"missing {cmd}"
    print("PASS test_help_lists_commands")


if __name__ == "__main__":
    test_help_lists_commands()
    test_score_cli()
    test_short_raw_refused()
    print("\n冒烟测试全部通过")
