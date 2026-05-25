#!/usr/bin/env bash
# =============================================================================
# eval_all.sh
# Run all evaluations for ONE strength level across all datasets,
# then print a paper-style results summary.
#
# Usage:
#   bash eval_all.sh [STRENGTH] [DATA_DIR] [REPO_ROOT]
#
# Defaults:
#   STRENGTH  = 0.75
#   DATA_DIR  = directory this script is called from
#   REPO_ROOT = AlphaSteer repo root (contains evaluation/, data/)
# =============================================================================

set -euo pipefail

STRENGTH="${1:-0.75}"
DATA_DIR="${2:-$(pwd)}"
REPO_ROOT="${3:-$(cd "$(dirname "$0")/.." && pwd)}"

EVAL_DIR="${REPO_ROOT}/evaluation"
SCRIPTS_DIR="${REPO_ROOT}/evaluation"

ALPACA_REF="${REPO_ROOT}/data/instructions/test/alpaca_eval.json"
ALPACA_CFG="${EVAL_DIR}/gpt4o-mini-config.yaml"
ALPACA_OUT="${DATA_DIR}/alpaca_eval_results"
TMP_DIR="${DATA_DIR}/.eval_tmp"
SUMMARY_FILE="${DATA_DIR}/results_summary_s${STRENGTH}.txt"

mkdir -p "${ALPACA_OUT}" "${TMP_DIR}"

echo "========================================================"
echo " AlphaSteer eval — strength = ${STRENGTH}"
echo " DATA_DIR  : ${DATA_DIR}"
echo " REPO_ROOT : ${REPO_ROOT}"
echo "========================================================"

# ── 1. Safety datasets (jailbreak.py) ─────────────────────────────────────────
echo ""
echo "── Safety datasets (jailbreak.py) ──────────────────────"

python "${SCRIPTS_DIR}/jailbreak.py" \
    --input-file  "${DATA_DIR}/aim_llama3.1_rfm_results.json" \
    --output-file "${DATA_DIR}/aim_llama3.1_rfm_results_eval.json" \
    --strength    "${STRENGTH}" \
    --model       gpt-4o-mini

python "${SCRIPTS_DIR}/jailbreak.py" \
    --input-file  "${DATA_DIR}/autodan_llama3.1_rfm_results.json" \
    --output-file "${DATA_DIR}/autodan_llama3.1_rfm_results_eval.json" \
    --strength    "${STRENGTH}" \
    --model       gpt-4o-mini

python "${SCRIPTS_DIR}/jailbreak.py" \
    --input-file  "${DATA_DIR}/cipher_llama3.1_rfm_results.json" \
    --output-file "${DATA_DIR}/cipher_llama3.1_rfm_results_eval.json" \
    --strength    "${STRENGTH}" \
    --model       gpt-4o-mini

python "${SCRIPTS_DIR}/jailbreak.py" \
    --input-file  "${DATA_DIR}/gcg_llama3.1_rfm_results.json" \
    --output-file "${DATA_DIR}/gcg_llama3.1_rfm_results_eval.json" \
    --strength    "${STRENGTH}" \
    --model       gpt-4o-mini

python "${SCRIPTS_DIR}/jailbreak.py" \
    --input-file  "${DATA_DIR}/jailbroken_llama3.1_rfm_results.json" \
    --output-file "${DATA_DIR}/jailbroken_llama3.1_rfm_results_eval.json" \
    --strength    "${STRENGTH}" \
    --model       gpt-4o-mini

python "${SCRIPTS_DIR}/jailbreak.py" \
    --input-file  "${DATA_DIR}/pair_llama3.1_rfm_results.json" \
    --output-file "${DATA_DIR}/pair_llama3.1_rfm_results_eval.json" \
    --strength    "${STRENGTH}" \
    --model       gpt-4o-mini

python "${SCRIPTS_DIR}/jailbreak.py" \
    --input-file  "${DATA_DIR}/renellm_llama3.1_rfm_results.json" \
    --output-file "${DATA_DIR}/renellm_llama3.1_rfm_results_eval.json" \
    --strength    "${STRENGTH}" \
    --model       gpt-4o-mini

# ── 2. XSTest (xstest.py) ─────────────────────────────────────────────────────
echo ""
echo "── XSTest (xstest.py) ──────────────────────────────────"

python "${SCRIPTS_DIR}/xstest.py" \
    --input_file      "${DATA_DIR}/xstest_llama3.1_rfm_results.json" \
    --question_column prompt \
    --model_name      "llama3.1_rfm" \
    --strength        "${STRENGTH}"

# ── 3. AlpacaEval ─────────────────────────────────────────────────────────────
echo ""
echo "── AlpacaEval (alpaca.py) ───────────────────────────────"

ALPACA_PREP="${TMP_DIR}/alpaca_eval_s${STRENGTH}.json"

python - <<PYEOF
import json

with open("${DATA_DIR}/alpaca_eval_llama3.1_rfm_results.json") as f:
    data = json.load(f)

out = []
for item in data:
    out.append({
        "instruction": item["instruction"],
        "output":      item["response_strength:${STRENGTH}"],
        "generator":   item.get("generator", "llama3.1_rfm"),
        "dataset":     item.get("dataset", ""),
    })

with open("${ALPACA_PREP}", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"[PREP] Wrote {len(out)} records → ${ALPACA_PREP}")
PYEOF

python "${EVAL_DIR}/alpaca.py" \
    --model-outputs     "${ALPACA_PREP}" \
    --reference-outputs "${ALPACA_REF}" \
    --annotators-config "${ALPACA_CFG}" \
    --name              "llama3.1_rfm_s${STRENGTH}" \
    --output-path       "${ALPACA_OUT}"

# ── 4. GSM8K & MATH500 (exact match) ──────────────────────────────────────────
echo ""
echo "── GSM8K / MATH500 (exact match) ───────────────────────"

python - <<PYEOF
import json, re, os

STRENGTH = "${STRENGTH}"
DATA_DIR = "${DATA_DIR}"

def extract_answer(text: str) -> str:
    if not text:
        return ""
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text.replace(',', ''))
    return nums[-1] if nums else ""

def normalize(s: str) -> str:
    return s.strip().replace(',', '').lower()

response_key = f"response_strength:{STRENGTH}"

for name, fname, answer_col in [
    ("gsm8k",  "gsm8k_llama3.1_rfm_results.json", "answer"),
    ("math500", "math_llama3.1_rfm_results.json",  "answer"),
]:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[SKIP] {fpath} not found")
        continue
    with open(fpath) as f:
        data = json.load(f)
    if not data or response_key not in data[0]:
        print(f"[SKIP] {name}: key '{response_key}' not found")
        continue
    correct = sum(
        normalize(extract_answer(item.get(response_key, ""))) == normalize(str(item.get(answer_col, "")))
        for item in data
    )
    print(f"[RESULT] {name}: {correct}/{len(data)} = {correct/len(data):.2%}  (strength={STRENGTH})")
PYEOF

# ── 5. Postprocessing — print paper-style summary ─────────────────────────────
echo ""
echo "── Results Summary (paper format) ──────────────────────"

python "${SCRIPTS_DIR}/summarize_results.py" \
    --data-dir   "${DATA_DIR}" \
    --strength   "${STRENGTH}" \
    --model-tag  "llama3.1_rfm" \
    --alpaca-out "${ALPACA_OUT}" \
    --save       "${SUMMARY_FILE}"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " All evaluations finished."
echo " Output files     → ${DATA_DIR}"
echo " AlpacaEval       → ${ALPACA_OUT}"
echo " Results summary  → ${SUMMARY_FILE}"
echo "========================================================"