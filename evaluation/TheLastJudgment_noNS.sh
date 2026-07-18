#!/usr/bin/env bash
# =============================================================================
# TheLastJudgment_noNS.sh - withoutNS variant
# Usage: bash TheLastJudgment_noNS.sh [STRENGTH] [DATA_DIR] [REPO_ROOT] [MODEL_NAME]
# MODEL_NAME default: llama3.1-70b  (used in file naming)
# =============================================================================

set -euo pipefail

STRENGTH="${1:-0.75}"
DATA_DIR="${2:-$(pwd)}"
REPO_ROOT="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_NAME="${4:-llama3.1-70b}"

EVAL_DIR="${REPO_ROOT}/evaluation"
ALPACA_REF="${REPO_ROOT}/data/instructions/test/alpaca_eval.json"
ALPACA_CFG="${EVAL_DIR}/gpt4o-mini-config.yaml"
ALPACA_OUT="${DATA_DIR}/alpaca_eval_results"
TMP_DIR="${DATA_DIR}/.eval_tmp"
SUMMARY_FILE="${DATA_DIR}/results_summary_s${STRENGTH}.txt"
MODEL_TAG="${MODEL_NAME}_rfm_no_nullspace"
FILE_PREFIX="${MODEL_NAME}_rfm_no_nullspace"

mkdir -p "${ALPACA_OUT}" "${TMP_DIR}"

ENV_FILE="${REPO_ROOT}/.env"
OPENAI_API_KEY_VAL=$(grep '^OPENAI_API_KEY=' "${ENV_FILE}" | tail -1 | cut -d'=' -f2-)

echo "========================================================"
echo " AlphaRFM (no NullSpace) eval — strength = ${STRENGTH}"
echo " DATA_DIR  : ${DATA_DIR}"
echo " MODEL     : ${MODEL_NAME}"
echo "========================================================"

# ── 1. Safety datasets — OpenRouter ───────────────────────────────────────────
echo ""
echo "── Safety datasets (jailbreak.py) ──────────────────────"
unset OPENAI_API_KEY

for DS in aim cipher jailbroken pair renellm; do
    INPUT="${DATA_DIR}/${DS}_${FILE_PREFIX}_results.json"
    OUTPUT="${DATA_DIR}/${DS}_${FILE_PREFIX}_results_eval.json"
    if [ -f "${INPUT}" ]; then
        python "${EVAL_DIR}/jailbreak.py" \
            --input-file  "${INPUT}" \
            --output-file "${OUTPUT}" \
            --strength    "${STRENGTH}" \
            --model       openai/gpt-4o-mini
    else
        echo "[SKIP] ${INPUT} not found"
    fi
done

# ── 2. XSTest — OpenRouter ────────────────────────────────────────────────────
echo ""
echo "── XSTest (xstest.py) ──────────────────────────────────"
python "${EVAL_DIR}/xstest.py" \
    --input_file      "${DATA_DIR}/xstest_${FILE_PREFIX}_results.json" \
    --question_column prompt \
    --model_name      "${MODEL_TAG}" \
    --strength        "${STRENGTH}"

# # ── 3. AlpacaEval — OpenAI ────────────────────────────────────────────────────
# echo ""
# echo "── AlpacaEval (alpaca.py) ───────────────────────────────"
# export OPENAI_API_KEY="${OPENAI_API_KEY_VAL}"

# ALPACA_SRC="${DATA_DIR}/alpaca_eval_${FILE_PREFIX}_results.json"
# ALPACA_PREP="${TMP_DIR}/alpaca_eval_nons_s${STRENGTH}.json"

# if [ -f "${ALPACA_SRC}" ]; then
#     python3 -c "
# import json, sys
# src = '${ALPACA_SRC}'
# out_path = '${ALPACA_PREP}'
# strength = '${STRENGTH}'
# with open(src) as f:
#     data = json.load(f)
# key = 'response_strength:' + strength
# out = [{'instruction': d['instruction'], 'output': d[key], 'generator': '${MODEL_TAG}', 'dataset': d.get('dataset','')} for d in data if key in d]
# with open(out_path, 'w') as f:
#     json.dump(out, f, indent=2, ensure_ascii=False)
# print(f'[PREP] Wrote {len(out)} records to ${ALPACA_PREP}')
# "

#     python "${EVAL_DIR}/alpaca.py" \
#         --model-outputs     "${ALPACA_PREP}" \
#         --reference-outputs "${ALPACA_REF}" \
#         --annotators-config "${ALPACA_CFG}" \
#         --name              "${MODEL_TAG}_s${STRENGTH}" \
#         --output-path       "${ALPACA_OUT}"
# else
#     echo "[SKIP] ${ALPACA_SRC} not found"
# fi

# ── 4. GSM8K & MATH500 ────────────────────────────────────────────────────────
echo ""
echo "── GSM8K / MATH500 (exact match) ───────────────────────"

python3 -c "
import json, re, os

STRENGTH = '${STRENGTH}'
DATA_DIR = '${DATA_DIR}'
FILE_PREFIX = '${FILE_PREFIX}'

def extract_gsm8k(text):
    if not text: return None
    m = re.search(r'####\s*([\d,.\-]+)', str(text))
    if m: return m.group(1).replace(',', '').strip()
    nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', str(text))
    return nums[-1].replace(',', '').strip() if nums else None

def extract_math500(text):
    if not text: return None
    m = re.search(r'\\\\boxed\{([^}]+)\}', str(text))
    if m: return m.group(1).strip()
    nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', str(text))
    return nums[-1].replace(',', '').strip() if nums else None

def norm_math(s):
    s = re.sub(r'\\\\(?:text|mathrm|mathbf|left|right)\{([^}]*)\}', r'\1', str(s).strip())
    return re.sub(r'\s+', '', s).replace(',', '').lower()

response_key = f'response_strength:{STRENGTH}'

for name, fname, extractor in [
    ('gsm8k',   f'gsm8k_{FILE_PREFIX}_results.json',  extract_gsm8k),
    ('math500', f'math_{FILE_PREFIX}_results.json',    extract_math500),
]:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f'[SKIP] {fpath} not found')
        continue
    with open(fpath) as f:
        data = json.load(f)
    if not data or response_key not in data[0]:
        print(f'[SKIP] {name}: key {response_key!r} not found')
        continue
    correct = total = 0
    for item in data:
        resp = item.get(response_key, '')
        gold = str(item.get('answer', ''))
        if not resp or not gold: continue
        total += 1
        pred = extractor(resp)
        gold_ans = extractor(gold) or gold
        if name == 'gsm8k':
            if pred == gold_ans: correct += 1
        else:
            if pred and norm_math(pred) == norm_math(gold_ans): correct += 1
    print(f'[RESULT] {name}: {correct}/{total} = {correct/total:.2%}  (strength={STRENGTH})')
"

# ── 5. Results Summary ────────────────────────────────────────────────────────
echo ""
echo "── Results Summary (paper format) ──────────────────────"
python "${EVAL_DIR}/summarize_results.py" \
    --data-dir   "${DATA_DIR}" \
    --strength   "${STRENGTH}" \
    --model-tag  "${MODEL_TAG}" \
    --alpaca-out "${ALPACA_OUT}" \
    --save       "${SUMMARY_FILE}"

echo ""
echo "========================================================"
echo " All evaluations finished."
echo " Output files  → ${DATA_DIR}"
echo " AlpacaEval    → ${ALPACA_OUT}"
echo " Summary       → ${SUMMARY_FILE}"
echo "========================================================"