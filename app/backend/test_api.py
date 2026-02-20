#!/usr/bin/env python3
"""Quick API tests - run with backend server at http://127.0.0.1:8000"""
import requests
import json

BASE = "http://127.0.0.1:8000"

def test(name, fn):
    try:
        fn()
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

def main():
    print("Testing Guardrails API...")
    ok = 0
    total = 0

    # 1. Health / guards list
    total += 1
    def _guards():
        r = requests.get(f"{BASE}/guards", timeout=5)
        r.raise_for_status()
        data = r.json()
        assert isinstance(data, list)
    if test("GET /guards", _guards): ok += 1

    # 2. Dataset preview - vectorAttacks
    total += 1
    def _preview():
        r = requests.post(f"{BASE}/dataset-preview", json={"dataset_name": "vectorAttacks", "max_samples": 10}, timeout=10)
        r.raise_for_status()
        data = r.json()
        assert "samples" in data
        assert len(data["samples"]) > 0
        assert "text" in data["samples"][0] or "prompt" in data["samples"][0]
    if test("POST /dataset-preview vectorAttacks", _preview): ok += 1

    # 3. Dataset preview - notInject
    total += 1
    def _preview_not():
        r = requests.post(f"{BASE}/dataset-preview", json={"dataset_name": "notInject", "max_samples": 5}, timeout=15)
        r.raise_for_status()
        data = r.json()
        assert "samples" in data
    if test("POST /dataset-preview notInject", _preview_not): ok += 1

    # 4. Evaluate (needs at least one guard)
    total += 1
    def _eval():
        r = requests.get(f"{BASE}/guards", timeout=5)
        guards = r.json()
        if not guards:
            raise Exception("No guards - create one first")
        gid = guards[0]["id"]
        r2 = requests.post(f"{BASE}/evaluate", json={
            "guard_id": gid,
            "dataset_name": "vectorAttacks",
            "max_samples": 5
        }, timeout=60)
        r2.raise_for_status()
        data = r2.json()
        assert "accuracy" in data
        assert "total_samples" in data
        assert data["total_samples"] > 0
    if test("POST /evaluate vectorAttacks (5 samples)", _eval): ok += 1

    # 5. Compare (needs 2 guards)
    total += 1
    def _compare():
        r = requests.get(f"{BASE}/guards", timeout=5)
        guards = r.json()
        if len(guards) < 2:
            raise Exception("Need 2 guards for compare")
        r2 = requests.post(f"{BASE}/compare", json={
            "guard_id_1": guards[0]["id"],
            "guard_id_2": guards[1]["id"],
            "dataset_name": "vectorAttacks",
            "max_samples": 5
        }, timeout=90)
        r2.raise_for_status()
        data = r2.json()
        assert "guard1" in data and "guard2" in data
        assert "differences" in data
    if test("POST /compare vectorAttacks", _compare): ok += 1

    # 6. Compare-all (small sample)
    total += 1
    def _compare_all():
        r = requests.get(f"{BASE}/guards", timeout=5)
        guards = r.json()
        if len(guards) < 2:
            raise Exception("Need 2 guards")
        r2 = requests.post(f"{BASE}/compare-all", json={
            "guard_id_1": guards[0]["id"],
            "guard_id_2": guards[1]["id"],
            "max_samples_per_dataset": 10
        }, timeout=120)
        r2.raise_for_status()
        data = r2.json()
        assert "results_by_dataset" in data
        assert len(data["results_by_dataset"]) >= 1
    if test("POST /compare-all", _compare_all): ok += 1

    # 7. Scan (single prompt)
    total += 1
    def _scan():
        r = requests.get(f"{BASE}/guards", timeout=5)
        guards = r.json()
        if not guards:
            raise Exception("No guards")
        r2 = requests.post(f"{BASE}/scan", json={
            "prompt": "What is the weather today?",
            "guard_id": guards[0]["id"]
        }, timeout=30)
        r2.raise_for_status()
        data = r2.json()
        assert "verdict" in data
        assert data["verdict"] in ("ALLOWED", "BLOCKED")
    if test("POST /scan", _scan): ok += 1

    print(f"\n{ok}/{total} tests passed")
    return 0 if ok == total else 1

if __name__ == "__main__":
    exit(main())
