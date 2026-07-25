#!/bin/bash
# run_extract_rc_hard_refusal.sh
# Phase B (activations only) for refusal_compliance_full_hard_refusal, run strictly AFTER
# run_build_rc_hard_refusal.sh finishes (never combined with build -- same phase-separation
# discipline as the original refusal_compliance_full pipeline). Same 8-shards-per-model,
# qwen2.5 -> llama3.1 -> gemma2 order.
#
# extract_refusal_compliance_embeddings.py's built-in --full flag only appends "_full" to the
# subdir name, not "_full_hard_refusal", so --data_dir/--output_dir are passed explicitly here.
set -uo pipefail
cd /home/workspace/mad_workspace/LLM/agopns_clean/AGOPNullSpace
mkdir -p logs/extract_rc_hard_refusal

NUM_SHARDS=8
MODELS=(qwen2.5 llama3.1 gemma2)

run_model_extract() {
    local model="$1"
    local data_dir="data/embeddings/${model}/refusal_compliance_full_hard_refusal"
    echo ""
    echo "================================================================"
    echo " EXTRACT (hard_refusal) WAVE: ${model}  ($(date +%H:%M:%S))"
    echo "================================================================"

    if [[ ! -f "${data_dir}/sorrybench_rc_dataset.json" ]]; then
        echo "!!! Missing ${data_dir}/sorrybench_rc_dataset.json -- did the build phase finish/merge?"
        return 1
    fi

    local pids=()
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
        logfile="logs/extract_rc_hard_refusal/${model}_shard${shard}.log"
        echo "[$(date +%H:%M:%S)] [${model} shard ${shard} -> cuda:${shard}] launching -> ${logfile}"
        python3 src/extract_refusal_compliance_embeddings.py \
            --model_name "${model}" --device "cuda:${shard}" \
            --data_dir "${data_dir}" --output_dir "${data_dir}" \
            --num_shards ${NUM_SHARDS} --shard_idx ${shard} \
            > "${logfile}" 2>&1 &
        pids+=($!)
    done

    for p in "${pids[@]}"; do wait "$p"; done

    echo "[$(date +%H:%M:%S)] ${model}: all ${NUM_SHARDS} shards finished, merging..."
    python3 src/extract_refusal_compliance_embeddings.py \
        --model_name "${model}" --data_dir "${data_dir}" --output_dir "${data_dir}" \
        --num_shards ${NUM_SHARDS} --merge \
        > "logs/extract_rc_hard_refusal/${model}_merge.log" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "!!! MERGE FAILED for ${model} -- see logs/extract_rc_hard_refusal/${model}_merge.log"
        return 1
    fi
    echo "=== EXTRACT ${model} DONE ($(date +%H:%M:%S)) ==="
    return 0
}

for model in "${MODELS[@]}"; do
    run_model_extract "$model"
    if [[ $? -ne 0 ]]; then
        echo "!!! ABORTING pipeline: ${model} extract failed."
        exit 1
    fi
done

echo ""
echo "ALL_EXTRACT_RC_HARD_REFUSAL_DONE"
