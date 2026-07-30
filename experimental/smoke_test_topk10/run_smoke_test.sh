#!/bin/bash
# One-off smoke test: 2-example sanity check for the topk10 ridge-combo
# steering matrices, all 10 datasets (excludes alpaca_eval) x 3 models.
# Everything (configs/inputs/outputs/logs) lives under experimental/ per
# repo convention -- production config/ and data/ dirs are untouched.
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root (AGOPNullSpace/)

BASE="experimental/smoke_test_topk10"
GPUS=(0 1 2 3 4 5 6)   # GPU 7 reserved for interactive use, never touched
DATASETS=(aim autodan cipher gcg gsm8k jailbroken math pair renellm xstest)
MODELS=(llama3.1 qwen2.5 gemma2)

JOBFILE=$(mktemp)
LOCKFILE=$(mktemp)
for model in "${MODELS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        echo "$BASE/configs/$model/$ds.yaml" >> "$JOBFILE"
    done
done

TOTAL=$(wc -l < "$JOBFILE")
echo "Queued $TOTAL smoke-test jobs across ${#GPUS[@]} GPUs (0-6)"

run_worker() {
    local gpu=$1
    while true; do
        local cfg=""
        exec 200>"$LOCKFILE"
        flock 200
        if [ -s "$JOBFILE" ]; then
            cfg=$(head -n1 "$JOBFILE")
            sed -i '1d' "$JOBFILE"
        fi
        flock -u 200
        exec 200>&-

        [ -z "$cfg" ] && break

        local tag
        tag=$(echo "$cfg" | sed "s|$BASE/configs/||; s|/|_|; s|\.yaml$||")
        echo "[$(date '+%H:%M:%S')] GPU $gpu: $cfg"

        CUDA_VISIBLE_DEVICES=$gpu python src/generate_response.py \
            --config_path "$cfg" >> "$BASE/logs/${tag}.log" 2>&1

        status=$?
        if [ $status -ne 0 ]; then
            echo "[$(date '+%H:%M:%S')] GPU $gpu: FAILED ($status) on $cfg — see $BASE/logs/${tag}.log"
        else
            echo "[$(date '+%H:%M:%S')] GPU $gpu: OK $cfg"
        fi
    done
    echo "[$(date '+%H:%M:%S')] GPU $gpu worker done"
}

for gpu in "${GPUS[@]}"; do
    run_worker "$gpu" &
done
wait

rm -f "$JOBFILE" "$LOCKFILE"
echo "Smoke test finished."
