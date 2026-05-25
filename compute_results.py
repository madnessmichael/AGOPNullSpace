#!/usr/bin/env python3
"""
Parallel LlamaGuard-4-12B evaluation runner
- 8 GPUs → 4 pairs (0,1 | 2,3 | 4,5 | 6,7)
- 10 jobs → rounds of 4, 4, 2
- Each subprocess sets CUDA_VISIBLE_DEVICES BEFORE importing torch/vllm
  so there's zero cross-GPU contamination and no core dumps.

Usage:
    python run_parallel.py

Remove or comment out these two lines in jailbreak_local.py first:
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""

import os
import sys
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# ── Config ────────────────────────────────────────────────────
HF_TOKEN   = "***REMOVED***"
MODEL      = "meta-llama/Llama-Guard-4-12B"
SCRIPT     = "evaluation/jailbreak_local.py"
BASE       = "./lovingpp/packPP-backup/withNS"
LOG_DIR    = "./eval_logs"
BATCH_SIZE = 1

os.makedirs(LOG_DIR, exist_ok=True)

# ── Job definitions ───────────────────────────────────────────
# (gpu_pair, input_file, log_tag)
JOBS = [
    # Round 1
    ("0,1", f"{BASE}/llama3.3-70b/aim_llama3.3-70b_rfm_results.json",          "33_aim"),
    ("2,3", f"{BASE}/llama3.3-70b/cipher_llama3.3-70b_rfm_results.json",        "33_cipher"),
    ("4,5", f"{BASE}/llama3.3-70b/jailbroken_llama3.3-70b_rfm_results.json",    "33_jailbroken"),
    ("6,7", f"{BASE}/llama3.3-70b/pair_llama3.3-70b_rfm_results.json",          "33_pair"),
    # Round 2
    ("0,1", f"{BASE}/llama3.3-70b/renellm_llama3.3-70b_rfm_results.json",       "33_renellm"),
    ("2,3", f"{BASE}/llama3.1-70b/aim_llama3.1-70b_rfm_results.json",           "31_aim"),
    ("4,5", f"{BASE}/llama3.1-70b/cipher_llama3.1-70b_rfm_results.json",        "31_cipher"),
    ("6,7", f"{BASE}/llama3.1-70b/jailbroken_llama3.1-70b_rfm_results.json",    "31_jailbroken"),
    # Round 3
    ("0,1", f"{BASE}/llama3.1-70b/pair_llama3.1-70b_rfm_results.json",          "31_pair"),
    ("2,3", f"{BASE}/llama3.1-70b/renellm_llama3.1-70b_rfm_results.json",       "31_renellm"),
]

# Group into rounds (max 4 parallel = 4 GPU pairs)
ROUNDS = [JOBS[0:4], JOBS[4:8], JOBS[8:10]]


def ts():
    return datetime.now().strftime("%H:%M:%S")


def run_job(gpus: str, input_file: str, tag: str) -> tuple[str, int]:
    """
    Spawns jailbreak_local.py as a subprocess with CUDA_VISIBLE_DEVICES set.
    Returns (tag, returncode).
    """
    log_path = os.path.join(LOG_DIR, f"{tag}.log")
    cmd = [
        sys.executable, SCRIPT,
        "--input-file",       input_file,
        "--llamaguard-model", MODEL,
        "--hf-token",         HF_TOKEN,
        "--batch-size",       str(BATCH_SIZE),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]  = gpus
    env["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"   # consistent ordering

    print(f"[{ts()}] ▶ START  gpu={gpus:<5}  {os.path.basename(input_file)}")

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    status = "✔ DONE " if proc.returncode == 0 else "✖ FAIL "
    print(f"[{ts()}] {status}  gpu={gpus:<5}  {os.path.basename(input_file)}"
          f"  (exit={proc.returncode})  log={log_path}")
    return tag, proc.returncode


def run_round(round_idx: int, jobs: list) -> list[tuple[str, int]]:
    print(f"\n{'═'*55}")
    print(f"  Round {round_idx}  —  {len(jobs)} jobs in parallel")
    print(f"{'═'*55}")
    t0 = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {ex.submit(run_job, g, f, t): t for g, f, t in jobs}
        for fut in as_completed(futures):
            results.append(fut.result())

    elapsed = time.time() - t0
    print(f"{'─'*55}")
    print(f"  Round {round_idx} done in {elapsed/60:.1f} min")
    return results


def main():
    all_results = []
    for i, round_jobs in enumerate(ROUNDS, start=1):
        all_results.extend(run_round(i, round_jobs))

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  SUMMARY")
    print(f"{'═'*55}")
    failed = [(tag, rc) for tag, rc in all_results if rc != 0]
    passed = [(tag, rc) for tag, rc in all_results if rc == 0]
    print(f"  ✔ Passed : {len(passed)}")
    print(f"  ✖ Failed : {len(failed)}")
    if failed:
        print("\n  Failed jobs:")
        for tag, rc in failed:
            print(f"    - {tag}  (exit={rc})  → {LOG_DIR}/{tag}.log")
    print(f"\n  All logs in: {LOG_DIR}/")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()