#!/bin/bash
# run_generation_rc.sh
# Orchestrates the llama3.1 + qwen2.5 refuse-compliance generation sweep
# (rc_ns + rc_nons, 10 datasets each = 40 jobs total). Gemma2 configs exist
# under config/gemma2_rc_{ns,nons}_rfm/ but are intentionally NOT run here.
#
# Each yaml's own `device: cuda:N` field (baked in by generate_configs_rc.sh)
# decides GPU placement. This script enforces AT MOST ONE running job per
# GPU index at any time -- never launches a second process on a GPU that
# already has one in flight, regardless of how generate_response.py's own
# internal strength loop paces itself.
set -uo pipefail

CONFIG_DIRS=(
    config/llama3.1_rc_ns_rfm
    config/llama3.1_rc_nons_rfm
    config/qwen2.5_rc_ns_rfm
    config/qwen2.5_rc_nons_rfm
)

mkdir -p logs/generate_rc
declare -A gpu_pid   # gpu index -> running PID (bash 4+ associative array)

wait_for_gpu() {
    local gpu="$1"
    local pid="${gpu_pid[$gpu]:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
    fi
}

n_total=0
for d in "${CONFIG_DIRS[@]}"; do
    for f in "$d"/*.yaml; do
        [[ -f "$f" ]] && n_total=$(( n_total + 1 ))
    done
done
echo "== ${n_total} generation jobs queued across ${#CONFIG_DIRS[@]} config dirs =="

n_done=0
for d in "${CONFIG_DIRS[@]}"; do
    for f in "$d"/*.yaml; do
        [[ -f "$f" ]] || continue
        gpu=$(grep -oE '^device: cuda:[0-9]+' "$f" | grep -oE '[0-9]+$')
        tag=$(basename "$d")_$(basename "$f" .yaml)
        logfile="logs/generate_rc/${tag}.log"

        wait_for_gpu "$gpu"

        echo "[$(date +%H:%M:%S)] launching ${f} on cuda:${gpu} -> ${logfile}"
        python3 src/generate_response.py --config_path "$f" > "$logfile" 2>&1 &
        gpu_pid[$gpu]=$!
        n_done=$(( n_done + 1 ))
        echo "   (${n_done}/${n_total} launched)"
    done
done

echo "All jobs launched -- waiting for the last one on each GPU to finish..."
for gpu in "${!gpu_pid[@]}"; do
    wait "${gpu_pid[$gpu]}"
done

echo "ALL_GENERATION_DONE"
