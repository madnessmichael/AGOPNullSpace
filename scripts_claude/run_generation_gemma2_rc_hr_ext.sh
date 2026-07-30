#!/bin/bash
# run_generation_gemma2_rc_hr_ext.sh
# gemma2, steering_matrix_gemma2_rfm_rc_full_hard_refusal.pt,
# EXTRA strengths {7.0,7.5,8.5,9.0,9.5,10.5,11.0}, all 10 datasets. GPU7 reserved.
# output_file is the SAME file as the (fixed-code) rc_hr regeneration --
# generate_response.py loads the existing output_file and merges new
# response_strength:{s} keys into it rather than overwriting.
set -uo pipefail
cd /home/workspace/mad_workspace/LLM/agopns_clean/AGOPNullSpace
mkdir -p logs/generate_rc

run_gpu_queue() {
    local gpu="$1"; shift
    for f in "$@"; do
        tag=$(basename "$(dirname "$f")")_$(basename "$f" .yaml)
        logfile="logs/generate_rc/${tag}.log"
        echo "[$(date +%H:%M:%S)] [gpu${gpu}] launching ${f} -> ${logfile}"
        if python3 src/generate_response.py --config_path "$f" > "$logfile" 2>&1; then
            echo "[$(date +%H:%M:%S)] [gpu${gpu}] finished ${f}"
        else
            echo "[$(date +%H:%M:%S)] [gpu${gpu}] !!! FAILED (exit $?) ${f} -- see ${logfile} !!!"
        fi
    done
}

run_gpu_queue 0 config/gemma2_rc_hr_ext_rfm/math.yaml &
p0=$!
run_gpu_queue 1 config/gemma2_rc_hr_ext_rfm/gsm8k.yaml &
p1=$!
run_gpu_queue 2 config/gemma2_rc_hr_ext_rfm/jailbroken.yaml &
p2=$!
run_gpu_queue 3 config/gemma2_rc_hr_ext_rfm/xstest.yaml config/gemma2_rc_hr_ext_rfm/aim.yaml &
p3=$!
run_gpu_queue 4 config/gemma2_rc_hr_ext_rfm/autodan.yaml config/gemma2_rc_hr_ext_rfm/cipher.yaml &
p4=$!
run_gpu_queue 5 config/gemma2_rc_hr_ext_rfm/gcg.yaml config/gemma2_rc_hr_ext_rfm/pair.yaml &
p5=$!
run_gpu_queue 6 config/gemma2_rc_hr_ext_rfm/renellm.yaml &
p6=$!

for p in $p0 $p1 $p2 $p3 $p4 $p5 $p6; do wait "$p"; done
echo "ALL_GEMMA2_RC_HR_EXT_DONE"
