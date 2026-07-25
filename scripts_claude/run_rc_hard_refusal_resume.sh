#!/bin/bash
# run_rc_hard_refusal_resume.sh
# Resume after the OpenAI daily RPD-limit incident: qwen2.5's build is already clean
# (0 judge errors) and kept as-is. Rebuild llama3.1 + gemma2 from scratch with the
# now-OpenRouter-backed hard_refusal_judge (separate quota, sidesteps the exhausted
# OpenAI gpt-4o-mini daily cap), then run the extract phase for all 3 models
# (extraction never started before the incident was caught).
set -uo pipefail
cd /home/workspace/mad_workspace/LLM/agopns_clean/AGOPNullSpace
mkdir -p logs/build_rc_hard_refusal

NUM_SHARDS=8

run_model_build() {
    local model="$1"
    echo ""
    echo "================================================================"
    echo " BUILD (hard_refusal, resumed) WAVE: ${model}  ($(date +%H:%M:%S))"
    echo "================================================================"

    local pids=()
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
        logfile="logs/build_rc_hard_refusal/${model}_shard${shard}_v2.log"
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
        > "logs/build_rc_hard_refusal/${model}_merge_v2.log" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "!!! MERGE FAILED for ${model} -- see logs/build_rc_hard_refusal/${model}_merge_v2.log"
        return 1
    fi

    # Sanity: this incident was caused by silent judge-error defaults, so verify
    # the fresh run actually has ~0 judge errors before moving on.
    n_err=$(python3 -c "import json; print(json.load(open('data/embeddings/${model}/refusal_compliance_full_hard_refusal/sorrybench_rc_report.json'))['judge_breakdown']['judge_errors'])")
    echo "[$(date +%H:%M:%S)] ${model}: judge_errors=${n_err}"
    if [[ "$n_err" -gt 50 ]]; then
        echo "!!! ${model}: judge_errors=${n_err} is suspiciously high -- ABORTING, inspect before continuing."
        return 1
    fi

    echo "=== BUILD ${model} DONE ($(date +%H:%M:%S)) ==="
    return 0
}

for model in llama3.1 gemma2; do
    run_model_build "$model"
    if [[ $? -ne 0 ]]; then
        echo "!!! ABORTING: ${model} build failed."
        exit 1
    fi
done

echo ""
echo "ALL_BUILD_RC_HARD_REFUSAL_RESUMED_DONE"

echo ""
echo "Starting extract phase for all 3 models (qwen2.5, llama3.1, gemma2)..."
bash scripts_claude/run_extract_rc_hard_refusal.sh
extract_rc=$?
if [[ $extract_rc -ne 0 ]]; then
    echo "!!! EXTRACT PHASE FAILED (exit ${extract_rc})."
    exit 1
fi

echo ""
echo "ALL_RC_HARD_REFUSAL_PIPELINE_DONE"
