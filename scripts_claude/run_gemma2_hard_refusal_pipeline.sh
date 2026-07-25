#!/bin/bash
# run_gemma2_hard_refusal_pipeline.sh
# gemma2-only build (Phase A) -> extract (Phase B) -> train RFM steering matrices for
# refusal_compliance_full_hard_refusal, deferred earlier while qwen2.5/llama3.1 were done.
# 8 shards per phase (one per GPU), same OpenRouter-backed hard_refusal_judge as qwen/llama.
set -uo pipefail
cd /home/workspace/mad_workspace/LLM/agopns_clean/AGOPNullSpace
mkdir -p logs/build_rc_hard_refusal logs/extract_rc_hard_refusal logs/calc_steering_matrix

NUM_SHARDS=8
MODEL=gemma2
DATA_DIR="data/embeddings/${MODEL}/refusal_compliance_full_hard_refusal"

# ---- Phase A: build ---------------------------------------------------------
echo ""
echo "================================================================"
echo " BUILD (hard_refusal) WAVE: ${MODEL}  ($(date +%H:%M:%S))"
echo "================================================================"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    logfile="logs/build_rc_hard_refusal/${MODEL}_shard${shard}_v2.log"
    echo "[$(date +%H:%M:%S)] [${MODEL} shard ${shard} -> cuda:${shard}] launching -> ${logfile}"
    python3 src/build_refusal_compliance_sorrybench.py \
        --model_name "${MODEL}" --device "cuda:${shard}" \
        --full --hard_refusal \
        --num_shards ${NUM_SHARDS} --shard_idx ${shard} \
        > "${logfile}" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "[$(date +%H:%M:%S)] ${MODEL}: all ${NUM_SHARDS} shards finished, merging..."
python3 src/build_refusal_compliance_sorrybench.py \
    --model_name "${MODEL}" --full --hard_refusal \
    --num_shards ${NUM_SHARDS} --merge \
    > "logs/build_rc_hard_refusal/${MODEL}_merge_v2.log" 2>&1
if [[ $? -ne 0 ]]; then
    echo "!!! MERGE FAILED for ${MODEL} -- see logs/build_rc_hard_refusal/${MODEL}_merge_v2.log"
    exit 1
fi

n_err=$(python3 -c "import json; print(json.load(open('${DATA_DIR}/sorrybench_rc_report.json'))['judge_breakdown']['judge_errors'])")
echo "[$(date +%H:%M:%S)] ${MODEL}: judge_errors=${n_err}"
if [[ "$n_err" -gt 50 ]]; then
    echo "!!! ${MODEL}: judge_errors=${n_err} is suspiciously high -- ABORTING, inspect before continuing."
    exit 1
fi
echo "=== BUILD ${MODEL} DONE ($(date +%H:%M:%S)) ==="

# ---- Phase B: extract --------------------------------------------------------
echo ""
echo "================================================================"
echo " EXTRACT (hard_refusal) WAVE: ${MODEL}  ($(date +%H:%M:%S))"
echo "================================================================"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    logfile="logs/extract_rc_hard_refusal/${MODEL}_shard${shard}.log"
    echo "[$(date +%H:%M:%S)] [${MODEL} shard ${shard} -> cuda:${shard}] launching -> ${logfile}"
    python3 src/extract_refusal_compliance_embeddings.py \
        --model_name "${MODEL}" --device "cuda:${shard}" \
        --data_dir "${DATA_DIR}" --output_dir "${DATA_DIR}" \
        --num_shards ${NUM_SHARDS} --shard_idx ${shard} \
        > "${logfile}" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "[$(date +%H:%M:%S)] ${MODEL}: all ${NUM_SHARDS} shards finished, merging..."
python3 src/extract_refusal_compliance_embeddings.py \
    --model_name "${MODEL}" --data_dir "${DATA_DIR}" --output_dir "${DATA_DIR}" \
    --num_shards ${NUM_SHARDS} --merge \
    > "logs/extract_rc_hard_refusal/${MODEL}_merge.log" 2>&1
if [[ $? -ne 0 ]]; then
    echo "!!! EXTRACT MERGE FAILED for ${MODEL} -- see logs/extract_rc_hard_refusal/${MODEL}_merge.log"
    exit 1
fi
echo "=== EXTRACT ${MODEL} DONE ($(date +%H:%M:%S)) ==="

# ---- Train RFM steering matrices (nullspace+gate, then no-nullspace) --------
echo ""
echo "================================================================"
echo " TRAIN steering matrices: ${MODEL}  ($(date +%H:%M:%S))"
echo "================================================================"

ns_out="data/steering_matrix/steering_matrix_${MODEL}_rfm_rc_full_hard_refusal.pt"
nons_out="data/steering_matrix/steering_matrix_${MODEL}_rfm_rc_full_hard_refusal_no_nullspace.pt"
r_path="data/steering_matrix/steering_matrix_${MODEL}_rfm_rc_full_hard_refusal_r.pt"

echo "[$(date +%H:%M:%S)] [${MODEL}] training nullspace+gate matrix -> ${ns_out}"
python3 src/calc_steering_matrix_rfm_rc.py \
    --model_name "${MODEL}" --embedding_dir "data/embeddings/${MODEL}" \
    --rc_subdir refusal_compliance_full_hard_refusal \
    --save_path "${ns_out}" --device "cuda:0" \
    > "logs/calc_steering_matrix/${MODEL}_rc_full_hard_refusal.log" 2>&1
if [[ $? -ne 0 ]]; then
    echo "!!! FAILED: ${MODEL} nullspace+gate matrix -- see logs/calc_steering_matrix/${MODEL}_rc_full_hard_refusal.log"
    exit 1
fi

echo "[$(date +%H:%M:%S)] [${MODEL}] training no-nullspace matrix -> ${nons_out}"
python3 src/calc_steering_matrix_rfm_rc_no_nullspace.py \
    --model_name "${MODEL}" --embedding_dir "data/embeddings/${MODEL}" \
    --refusal_vectors_path "${r_path}" \
    --save_path "${nons_out}" --device "cuda:0" \
    > "logs/calc_steering_matrix/${MODEL}_rc_full_hard_refusal_no_nullspace.log" 2>&1
if [[ $? -ne 0 ]]; then
    echo "!!! FAILED: ${MODEL} no-nullspace matrix -- see logs/calc_steering_matrix/${MODEL}_rc_full_hard_refusal_no_nullspace.log"
    exit 1
fi

echo "=== TRAIN ${MODEL} DONE ($(date +%H:%M:%S)) ==="
echo ""
echo "ALL_GEMMA2_HARD_REFUSAL_PIPELINE_DONE"
