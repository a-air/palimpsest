#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""状态机测试：崩溃恢复、失败晋级、锁。"""
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refiner as R


def make_world(n=4, good=True):
    root = Path(tempfile.mkdtemp(prefix="sm-"))
    for sub in ("archive", "cold", "state"):
        (root / sub).mkdir()
    R.ROOT, R.ARCHIVE, R.COLD, R.STATE = root, root / "archive", root / "cold", root / "state"
    R.LIFECYCLE = root / "state" / "lifecycle.json"
    R.LOCK = root / "state" / "lifecycle.lock"
    R.FACTS = root / "state" / "facts.jsonl"

    raw = "3月2日与供应商谈定交付。3月5日改期到3月18日。3月8日客户确认。"
    fact = ("3月2日谈定交付；3月5日改期到3月18日；3月8日客户确认。" if good
            else "挺好。顺利。没问题。")
    for i in range(n):
        d = root / "archive" / f"2026-04-0{i+1}"
        d.mkdir()
        (d / f"{i}_raw.md").write_text(raw, encoding="utf-8")
        (d / f"{i}_node.json").write_text(json.dumps({
            "node_title": f"批次{i}", "summary": "交付改期",
            "facts": [{"text": fact, "kind": "fact", "confidence": "S"}],
            "persona_anchors": [], "open_threads": [],
        }, ensure_ascii=False), encoding="utf-8")
    return root


def test_recover_half_moved_pack():
    """模拟"移到一半断电"：2 份已在冷箱，2 份还在 archive，状态仍是 packing。"""
    root = make_world(4)
    data = {"batches": [], "packs": []}
    pack = root / "cold" / "P1"
    pack.mkdir()
    for i in range(4):
        rel = f"archive/2026-04-0{i+1}"
        data["batches"].append({"dir": rel, "state": "packing",
                                "plan": {"pack": "P1", "pack_dir": "cold/P1"}})
    # 前两份"已搬走"
    import shutil
    for i in range(2):
        shutil.move(str(root / f"archive/2026-04-0{i+1}"), str(pack / f"2026-04-0{i+1}"))
    R.save_lifecycle(data)

    notes = R.recover_packing(R.load_lifecycle())
    assert any("cold" in n for n in notes) and any("sealed" in n for n in notes), notes
    data2 = R.load_lifecycle()
    states = [b["state"] for b in data2["batches"]]
    assert states[:2] == ["cold", "cold"], states
    assert states[2:] == ["sealed", "sealed"], states
    print("PASS test_recover_half_moved_pack", states)


def test_escalation_to_quarantine():
    """连续 3 次不过 → 隔离区；且过程中不抛异常。"""
    make_world(4, good=False)
    for _ in range(4):
        R.cmd_seal(SimpleNamespace(yes=True))
    for _ in range(3):
        code = R.cmd_review(SimpleNamespace(pass_score=99))
        assert code == 1
    data = R.load_lifecycle()
    states = [b["state"] for b in data["batches"]]
    assert all(s == "quarantine" for s in states), states
    assert all(b.get("attempts") == 3 for b in data["batches"])
    print("PASS test_escalation_to_quarantine", states)


def test_fifo_no_starvation():
    """失败的旧批次必须优先被再次抽到，而不是被新批次挤出窗口。"""
    make_world(4, good=False)
    for _ in range(4):
        R.cmd_seal(SimpleNamespace(yes=True))
    R.cmd_review(SimpleNamespace(pass_score=99))
    old_dirs = [b["dir"] for b in R.load_lifecycle()["batches"]]

    # 再来 4 份新封存
    for i in range(4):
        d = R.ARCHIVE / f"2026-05-0{i+1}"
        d.mkdir()
        (d / "x_raw.md").write_text("3月2日谈定。", encoding="utf-8")
        (d / "x_node.json").write_text(json.dumps({"facts": [{"text": "3月2日谈定。"}]}), encoding="utf-8")
    for _ in range(4):
        R.cmd_seal(SimpleNamespace(yes=True))
    R.cmd_review(SimpleNamespace(pass_score=99))

    data = R.load_lifecycle()
    retried = [b for b in data["batches"] if b["dir"] in old_dirs and b.get("attempts", 0) >= 2]
    assert retried, "旧批次被饿死了，FIFO 没生效"
    print("PASS test_fifo_no_starvation", len(retried))


def test_lock_blocks_second_process():
    make_world(1)
    ok1, _ = R.acquire_lock()
    assert ok1
    ok2, why = R.acquire_lock()
    assert not ok2 and "另一个进程" in why, why
    R.release_lock()
    ok3, _ = R.acquire_lock()
    assert ok3
    R.release_lock()
    print("PASS test_lock_blocks_second_process")


def test_stale_lock_can_be_taken():
    root = make_world(1)
    R.LOCK.write_text(json.dumps({"pid": 1, "ts": time.time() - R.LOCK_STALE_SEC - 5}), encoding="utf-8")
    ok, why = R.acquire_lock()
    assert ok, why
    R.release_lock()
    print("PASS test_stale_lock_can_be_taken")


if __name__ == "__main__":
    test_recover_half_moved_pack()
    test_escalation_to_quarantine()
    test_fifo_no_starvation()
    test_lock_blocks_second_process()
    test_stale_lock_can_be_taken()
    print("\n状态机测试全部通过")
