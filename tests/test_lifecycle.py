#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生命周期测试：热 → 封存 → 核验 → 冷箱。"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refiner as R


def make_world(n_batches=4, good=True):
    root = Path(tempfile.mkdtemp(prefix="lifecycle-"))
    (root / "archive").mkdir()
    (root / "cold").mkdir()
    (root / "state").mkdir()
    R.ROOT, R.ARCHIVE, R.COLD, R.STATE = root, root / "archive", root / "cold", root / "state"
    R.LIFECYCLE = root / "state" / "lifecycle.json"
    R.FACTS = root / "state" / "facts.jsonl"

    raw = "3月2日与供应商谈定交付。3月5日改期到3月18日。3月8日客户确认。"
    fact = ("3月2日谈定交付；3月5日改期到3月18日；3月8日客户确认。" if good
            else "供应商很好。一切都顺利。没有问题。")
    for i in range(n_batches):
        d = root / "archive" / f"2026-03-0{i+1}"
        d.mkdir()
        (d / f"{i}_raw.md").write_text(raw, encoding="utf-8")
        (d / f"{i}_node.json").write_text(json.dumps({
            "node_title": f"批次{i}", "summary": "交付改期",
            "facts": [{"text": fact, "kind": "fact", "confidence": "S"}],
            "persona_anchors": [], "open_threads": [],
        }, ensure_ascii=False), encoding="utf-8")
        (d / f"{i}_meta.json").write_text(json.dumps({"score": 60}), encoding="utf-8")
    return root


def test_seal_needs_four_before_review():
    make_world(4)
    for _ in range(3):
        assert R.cmd_seal(SimpleNamespace(yes=True)) == 0
    assert R.cmd_review(SimpleNamespace(pass_score=40)) == 1, "不足 4 份不该能核验"
    print("PASS test_seal_needs_four_before_review")


def test_review_pass_moves_to_cold():
    root = make_world(4, good=True)
    for _ in range(4):
        R.cmd_seal(SimpleNamespace(yes=True))
    code = R.cmd_review(SimpleNamespace(pass_score=40))
    assert code == 0, "该通过却没通过"
    packs = list((root / "cold").iterdir())
    assert len(packs) == 1, packs
    moved = list(packs[0].rglob("*_node.json"))
    assert len(moved) == 4, f"冷箱里应打包 4 份，实际 {len(moved)}"
    assert not list((root / "archive").glob("2026-03-0*")), "原批次应已移走"
    data = R.load_lifecycle()
    assert all(b["state"] == "cold" for b in data["batches"])
    print("PASS test_review_pass_moves_to_cold", packs[0].name)


def test_review_fail_keeps_sealed():
    root = make_world(4, good=False)
    for _ in range(4):
        R.cmd_seal(SimpleNamespace(yes=True))
    code = R.cmd_review(SimpleNamespace(pass_score=99))
    assert code == 1, "分数不够却通过了"
    assert not list((root / "cold").iterdir()), "不该打包"
    data = R.load_lifecycle()
    assert all(b["state"] == "sealed" for b in data["batches"])
    print("PASS test_review_fail_keeps_sealed")


if __name__ == "__main__":
    test_seal_needs_four_before_review()
    test_review_pass_moves_to_cold()
    test_review_fail_keeps_sealed()
    print("\n生命周期测试全部通过")
