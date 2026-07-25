#!/bin/bash
# run_build_rc_hard_refusal.sh
# Phase A (build+judge only, no activations) for the new refusal_compliance_full_hard_refusal
# dataset: same 9,236-row full SORRY-Bench set as refusal_compliance_full, but judged with the
# stricter hard-refusal criterion (evaluation/hard_refusal_judge.py) instead of the harm-content
# jailbreak judge. 8 shards per model (one per GPU), models run sequentially in the order
# qwen2.5 -> llama3.1 -> gemma2, exactly like the original refusal_compliance_full build.
set -uo pipefail
cd /home/workspace/mad_workspace/LLM/agopns_clean/AGOPNullSpace
mkdir -p logs/build_rc_hard_refusal

NUM_SHARDS=8
MODELS=(qwen2.5 llama3.1 gemma2)

run_model_build() {
    local model="$1"
    echo ""
    echo "================================================================"
    echo " BUILD (hard_refusal) WAVE: ${model}  ($(date +%H:%M:%S))"
    echo "================================================================"

    local pids=()
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
        logfile="logs/build_rc_hard_refusal/${model}_shard${shard}.log"
        echo "[$(date +%H:%M:%S)] [${model} shard ${shard} -> cuda:${shard}] launching -> ${logfile}"
        python3 src/build_refusal_compliance_sorrybench.py \
            --model_name "${model}" --device "cuda:${shard}" \
            --full --hard_refusal \
            --num_shards ${NUM_SHARDS} --shard_idx ${shard} \
            > "${logfile}" 2>&1 &
        pids+=($!)
    done

    for p in "${pids[@]}"; do wait "$p"; done

    echo "[$(date +%H:%M:%S)] ${model}: all ${NUM_SHARDS} shards finished, merging..."
    python3 src/build_refusal_compliance_sorrybench.py \
        --model_name "${model}" --full --hard_refusal \
        --num_shards ${NUM_SHARDS} --merge \
        > "logs/build_rc_hard_refusal/${model}_merge.log" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "!!! MERGE FAILED for ${model} -- see logs/build_rc_hard_refusal/${model}_merge.log"
        return 1
    fi
    echo "=== BUILD ${model} DONE ($(date +%H:%M:%S)) ==="
    return 0
}

for model in "${MODELS[@]}"; do
    run_model_build "$model"
    if [[ $? -ne 0 ]]; then
        echo "!!! ABORTING pipeline: ${model} build failed."
        exit 1
    fi
done

echo ""
echo "ALL_BUILD_RC_HARD_REFUSAL_DONE"
