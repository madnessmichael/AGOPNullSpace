#!/bin/bash
# run_extract_and_train_qwen_llama_hard_refusal.sh
# gemma2 build is deferred per user request -- this only extracts embeddings and
# trains RFM steering matrices for qwen2.5 + llama3.1 on refusal_compliance_full_hard_refusal.
# Order: qwen2.5 -> llama3.1 for both extract and train, matching the established convention.
set -uo pipefail
cd /home/workspace/mad_workspace/LLM/agopns_clean/AGOPNullSpace
mkdir -p logs/extract_rc_hard_refusal

NUM_SHARDS=8
MODELS=(qwen2.5 llama3.1)

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
        echo "!!! ABORTING: ${model} extract failed."
        exit 1
    fi
done

echo ""
echo "ALL_EXTRACT_QWEN_LLAMA_HARD_REFUSAL_DONE"

# ---- Train RFM steering matrices (nullspace+gate, then no-nullspace) --------
mkdir -p logs/calc_steering_matrix
echo ""
echo "================================================================"
echo " TRAIN steering matrices: qwen2.5 -> llama3.1  ($(date +%H:%M:%S))"
echo "================================================================"

train_one() {
    local model="$1"; local gpu="$2"
    local ns_out="data/steering_matrix/steering_matrix_${model}_rfm_rc_full_hard_refusal.pt"
    local nons_out="data/steering_matrix/steering_matrix_${model}_rfm_rc_full_hard_refusal_no_nullspace.pt"
    local r_path="data/steering_matrix/steering_matrix_${model}_rfm_rc_full_hard_refusal_r.pt"

    echo "[$(date +%H:%M:%S)] [${model}] training nullspace+gate matrix -> ${ns_out}"
    python3 src/calc_steering_matrix_rfm_rc.py \
        --model_name "${model}" --embedding_dir "data/embeddings/${model}" \
        --rc_subdir refusal_compliance_full_hard_refusal \
        --save_path "${ns_out}" --device "cuda:${gpu}" \
        > "logs/calc_steering_matrix/${model}_rc_full_hard_refusal.log" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "!!! FAILED: ${model} nullspace+gate matrix -- see logs/calc_steering_matrix/${model}_rc_full_hard_refusal.log"
        return 1
    fi

    echo "[$(date +%H:%M:%S)] [${model}] training no-nullspace matrix -> ${nons_out}"
    python3 src/calc_steering_matrix_rfm_rc_no_nullspace.py \
        --model_name "${model}" --embedding_dir "data/embeddings/${model}" \
        --refusal_vectors_path "${r_path}" \
        --save_path "${nons_out}" --device "cuda:${gpu}" \
        > "logs/calc_steering_matrix/${model}_rc_full_hard_refusal_no_nullspace.log" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "!!! FAILED: ${model} no-nullspace matrix -- see logs/calc_steering_matrix/${model}_rc_full_hard_refusal_no_nullspace.log"
        return 1
    fi

    echo "=== TRAIN ${model} DONE ($(date +%H:%M:%S)) ==="
    return 0
}

train_one qwen2.5 0
if [[ $? -ne 0 ]]; then echo "!!! ABORTING: qwen2.5 training failed."; exit 1; fi

train_one llama3.1 0
if [[ $? -ne 0 ]]; then echo "!!! ABORTING: llama3.1 training failed."; exit 1; fi

echo ""
echo "ALL_TRAIN_QWEN_LLAMA_HARD_REFUSAL_DONE"
