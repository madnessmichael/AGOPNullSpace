#!/bin/bash
# run_generation_rc_full.sh
# Runs the refusal_compliance_full sweep (15 combined strengths: 0.0-1.5 + 8.0-11.0)
# in three sequential waves -- qwen2.5, then llama3.1, then gemma2 -- as requested.
# Within each wave, jobs are grouped by the `device: cuda:N` already baked into
# each yaml by generate_configs_rc_full.sh, and each GPU runs its own jobs
# strictly one-at-a-time via an independent background queue (never more than
# one process per GPU at once, no cross-GPU blocking).
set -uo pipefail
cd /home/workspace/mad_workspace/LLM/agopns_clean/AGOPNullSpace
mkdir -p logs/generate_rc

run_gpu_queue() {
    local gpu="$1"; shift
    for f in "$@"; do
        tag=$(basename "$(dirname "$f")")_$(basename "$f" .yaml)
        logfile="logs/generate_rc/${tag}.log"
        echo "[$(date +%H:%M:%S)] [gpu${gpu}] launching ${f} -> ${logfile}"
        python3 src/generate_response.py --config_path "$f" > "$logfile" 2>&1
        echo "[$(date +%H:%M:%S)] [gpu${gpu}] finished ${f}"
    done
}

run_model_wave() {
    local model="$1"
    echo ""
    echo "================================================================"
    echo " WAVE: ${model}  ($(date +%H:%M:%S))"
    echo "================================================================"

    local all_yamls
    all_yamls=$(ls config/${model}_rc_ns_full_rfm/*.yaml config/${model}_rc_nons_full_rfm/*.yaml 2>/dev/null)

    # Bucket yaml files by their device: cuda:N field.
    declare -A buckets
    for f in $all_yamls; do
        dev=$(grep -m1 "^device:" "$f" | sed 's/device: cuda://')
        buckets[$dev]="${buckets[$dev]:-} $f"
    done

    local pids=()
    for gpu in "${!buckets[@]}"; do
        run_gpu_queue "$gpu" ${buckets[$gpu]} &
        pids+=($!)
    done

    for p in "${pids[@]}"; do wait "$p"; done
    echo "=== WAVE ${model} DONE ($(date +%H:%M:%S)) ==="
}

run_model_wave qwen2.5
run_model_wave llama3.1
run_model_wave gemma2

echo ""
echo "ALL_RC_FULL_SWEEP_DONE"
