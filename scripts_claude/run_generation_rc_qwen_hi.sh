#!/bin/bash
# run_generation_rc_qwen_hi.sh
# Orchestrates the qwen2.5 high-strength (8.0-11.0) follow-up sweep: rc_ns_hi +
# rc_nons_hi, 10 datasets each = 20 jobs. Same 1-job-per-GPU discipline as
# run_generation_rc.sh.
set -uo pipefail

CONFIG_DIRS=(
    config/qwen2.5_rc_ns_hi_rfm
    config/qwen2.5_rc_nons_hi_rfm
)

mkdir -p logs/generate_rc
declare -A gpu_pid

wait_for_gpu() {
    local gpu="$1"
    local pid="${gpu_pid[$gpu]:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
    fi
}

n_total=0
for d in "${CONFIG_DIRS[@]}"; do
    for f in "$d"/*.yaml; do [[ -f "$f" ]] && n_total=$(( n_total + 1 )); done
done
echo "== ${n_total} qwen2.5 high-strength jobs queued =="

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

echo "All jobs launched -- waiting for the last one on each GPU..."
for gpu in "${!gpu_pid[@]}"; do
    wait "${gpu_pid[$gpu]}"
done
echo "ALL_QWEN_HI_DONE"
