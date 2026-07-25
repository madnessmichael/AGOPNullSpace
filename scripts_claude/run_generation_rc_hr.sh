#!/bin/bash
# run_generation_rc_hr.sh
# qwen2.5 only, steering_matrix_qwen2.5_rfm_rc_full_hard_refusal.pt (nullspace+gate),
# strengths {0.0, 10.0}, all 10 datasets. GPU7 reserved (active user process).
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

run_gpu_queue 0 config/qwen2.5_rc_hr_rfm/aim.yaml config/qwen2.5_rc_hr_rfm/gsm8k.yaml &
p0=$!
run_gpu_queue 1 config/qwen2.5_rc_hr_rfm/autodan.yaml config/qwen2.5_rc_hr_rfm/math.yaml &
p1=$!
run_gpu_queue 2 config/qwen2.5_rc_hr_rfm/cipher.yaml config/qwen2.5_rc_hr_rfm/xstest.yaml &
p2=$!
run_gpu_queue 3 config/qwen2.5_rc_hr_rfm/gcg.yaml &
p3=$!
run_gpu_queue 4 config/qwen2.5_rc_hr_rfm/jailbroken.yaml &
p4=$!
run_gpu_queue 5 config/qwen2.5_rc_hr_rfm/pair.yaml &
p5=$!
run_gpu_queue 6 config/qwen2.5_rc_hr_rfm/renellm.yaml &
p6=$!

for p in $p0 $p1 $p2 $p3 $p4 $p5 $p6; do wait "$p"; done
echo "ALL_QWEN_RC_HR_DONE"
