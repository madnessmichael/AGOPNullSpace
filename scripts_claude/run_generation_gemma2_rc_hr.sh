#!/bin/bash
# run_generation_gemma2_rc_hr.sh
# gemma2, steering_matrix_gemma2_rfm_rc_full_hard_refusal.pt,
# strengths {0.0,0.5,1.0,5.0,8.0,10.0,10.5,11.0}, all 10 datasets. GPU7 reserved.
# math (4096 tok) and gsm8k (2048 tok) isolated alone -- much heavier than the
# 128-tok attack datasets at batch_size=1.
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

run_gpu_queue 0 config/gemma2_rc_hr_rfm/math.yaml &
p0=$!
run_gpu_queue 1 config/gemma2_rc_hr_rfm/gsm8k.yaml &
p1=$!
run_gpu_queue 2 config/gemma2_rc_hr_rfm/jailbroken.yaml &
p2=$!
run_gpu_queue 3 config/gemma2_rc_hr_rfm/xstest.yaml config/gemma2_rc_hr_rfm/aim.yaml &
p3=$!
run_gpu_queue 4 config/gemma2_rc_hr_rfm/autodan.yaml config/gemma2_rc_hr_rfm/cipher.yaml &
p4=$!
run_gpu_queue 5 config/gemma2_rc_hr_rfm/gcg.yaml config/gemma2_rc_hr_rfm/pair.yaml &
p5=$!
run_gpu_queue 6 config/gemma2_rc_hr_rfm/renellm.yaml &
p6=$!

for p in $p0 $p1 $p2 $p3 $p4 $p5 $p6; do wait "$p"; done
echo "ALL_GEMMA2_RC_HR_DONE"
