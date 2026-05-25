#!/usr/bin/env bash
# =============================================================================
# TheLastJudgment_noNS.sh
# Same as TheLastJudgment.sh but for withoutNS variant
# (file naming: *_no_nullspace_* instead of *_rfm_*)
#
# Usage:
#   bash TheLastJudgment_noNS.sh [STRENGTH] [DATA_DIR] [REPO_ROOT]
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
echo " AlphaRFM (no NullSpace) eval — strength = ${STRENGTH}"
echo " DATA_DIR  : ${DATA_DIR}"
echo " REPO_ROOT : ${REPO_ROOT}"
echo "========================================================"

# ── 1. Safety datasets (jailbreak.py) ─────────────────────────────────────────
echo ""
echo "── Safety datasets (jailbreak.py) ──────────────────────"

for DS in aim autodan cipher gcg jailbroken pair renellm; do
    python "${SCRIPTS_DIR}/jailbreak.py" \
        --input-file  "${DATA_DIR}/${DS}_llama3.1_rfm_no_nullspace_results.json" \
        --output-file "${DATA_DIR}/${DS}_llama3.1_rfm_no_nullspace_results_eval.json" \
        --strength    "${STRENGTH}" \
        --model       openai/gpt-4o-mini
done

# ── 2. XSTest (xstest.py) ─────────────────────────────────────────────────────
echo ""
echo "── XSTest (xstest.py) ──────────────────────────────────"

python "${SCRIPTS_DIR}/xstest.py" \
    --input_file      "${DATA_DIR}/xstest_llama3.1_rfm_no_nullspace_results.json" \
    --question_column prompt \
    --model_name      "llama3.1_rfm_no_nullspace" \
    --strength        "${STRENGTH}"

# ── 3. AlpacaEval ─────────────────────────────────────────────────────────────
echo ""
echo "── AlpacaEval (alpaca.py) ───────────────────────────────"

ALPACA_PREP="${TMP_DIR}/alpaca_eval_nons_s${STRENGTH}.json"

python - <<PYEOF
import json

with open("${DATA_DIR}/alpaca_eval_llama3.1_rfm_no_nullspace_results.json") as f:
    data = json.load(f)

out = []
for item in data:
    out.append({
        "instruction": item["instruction"],
        "output":      item["response_strength:${STRENGTH}"],
        "generator":   "llama3.1_rfm_no_nullspace",
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
    --name              "llama3.1_rfm_no_nullspace_s${STRENGTH}" \
    --output-path       "${ALPACA_OUT}"

# ── 4. GSM8K & MATH500 (exact match) ──────────────────────────────────────────
echo ""
echo "── GSM8K / MATH500 (exact match) ───────────────────────"

python - <<PYEOF
import json, re, os

STRENGTH = "${STRENGTH}"
DATA_DIR = "${DATA_DIR}"

def extract_gsm8k(text):
    if not text: return None
    m = re.search(r'####\s*([\d,\.\-]+)', str(text))
    if m: return m.group(1).replace(',', '').strip()
    nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', str(text))
    return nums[-1].replace(',', '').strip() if nums else None

def extract_math500(text):
    if not text: return None
    m = re.search(r'\\boxed\{([^}]+)\}', str(text))
    if m: return m.group(1).strip()
    m = re.search(r'(?:answer is|=)\s*([\d\w\+\-\*/\^\(\)\.]+)', str(text), re.I)
    if m: return m.group(1).strip()
    nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', str(text))
    return nums[-1].replace(',', '').strip() if nums else None

def norm_math(s):
    s = re.sub(r'\\(text|mathrm|mathbf|left|right)\{([^}]*)\}', r'\2', str(s).strip())
    return re.sub(r'\s+', '', s).replace(',', '').lower()

response_key = f"response_strength:{STRENGTH}"

for name, fname, extractor, answer_col in [
    ("gsm8k",   "gsm8k_llama3.1_rfm_no_nullspace_results.json",  extract_gsm8k,   "answer"),
    ("math500", "math_llama3.1_rfm_no_nullspace_results.json",    extract_math500, "answer"),
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
    correct = total = 0
    for item in data:
        resp = item.get(response_key, "")
        gold = str(item.get(answer_col, ""))
        if not resp or not gold: continue
        total += 1
        pred = extractor(resp)
        gold_ans = extractor(gold) or gold
        if name == "gsm8k":
            if pred == gold_ans: correct += 1
        else:
            if pred and norm_math(pred) == norm_math(gold_ans): correct += 1
    print(f"[RESULT] {name}: {correct}/{total} = {correct/total:.2%}  (strength={STRENGTH})")
PYEOF

# ── 5. Results Summary ────────────────────────────────────────────────────────
echo ""
echo "── Results Summary (paper format) ──────────────────────"

python "${SCRIPTS_DIR}/summarize_results.py" \
    --data-dir   "${DATA_DIR}" \
    --strength   "${STRENGTH}" \
    --model-tag  "llama3.1_rfm_no_nullspace" \
    --alpaca-out "${ALPACA_OUT}" \
    --save       "${SUMMARY_FILE}"

echo ""
echo "========================================================"
echo " All evaluations finished."
echo " Output files     → ${DATA_DIR}"
echo " AlpacaEval       → ${ALPACA_OUT}"
echo " Results summary  → ${SUMMARY_FILE}"
echo "========================================================"