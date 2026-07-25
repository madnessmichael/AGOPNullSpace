#!/bin/bash
# run_generation_rc_hr2.sh
# qwen2.5, steering_matrix_qwen2.5_rfm_rc_full_hard_refusal.pt, strengths {10.5, 11.0, 11.5},
# all 10 datasets. GPU7 reserved. math isolated alone (slowest job last time, ~27min for 2 strengths).
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

run_gpu_queue 0 config/qwen2.5_rc_hr2_rfm/math.yaml &
p0=$!
run_gpu_queue 1 config/qwen2.5_rc_hr2_rfm/cipher.yaml config/qwen2.5_rc_hr2_rfm/aim.yaml &
p1=$!
run_gpu_queue 2 config/qwen2.5_rc_hr2_rfm/xstest.yaml config/qwen2.5_rc_hr2_rfm/autodan.yaml &
p2=$!
run_gpu_queue 3 config/qwen2.5_rc_hr2_rfm/gcg.yaml &
p3=$!
run_gpu_queue 4 config/qwen2.5_rc_hr2_rfm/jailbroken.yaml &
p4=$!
run_gpu_queue 5 config/qwen2.5_rc_hr2_rfm/pair.yaml &
p5=$!
run_gpu_queue 6 config/qwen2.5_rc_hr2_rfm/renellm.yaml config/qwen2.5_rc_hr2_rfm/gsm8k.yaml &
p6=$!

for p in $p0 $p1 $p2 $p3 $p4 $p5 $p6; do wait "$p"; done
echo "ALL_QWEN_RC_HR2_DONE"
