#!/bin/bash
# run_generation_llama_dim_rc_hr.sh
# llama3.1, steering_matrix_llama3.1_dim_rc_full_hard_refusal.pt (DIM ablation,
# same rc_hr refuse-compliance data as the RFM production matrix, concept
# vector r replaced by DiffMean instead of RFM/AGOP).
# All 10 rc_hr datasets EXCEPT math (left for later, it's slow). Output files
# are NEW (*_dim_rc_hr.json), do not touch the existing *_rc_hr.json RFM results.
# GPU7 reserved for the user's own interactive notebook.
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

run_gpu_queue 1 config/llama3.1_dim_rc_hr_rfm/gsm8k.yaml &
p1=$!
run_gpu_queue 2 config/llama3.1_dim_rc_hr_rfm/jailbroken.yaml &
p2=$!
run_gpu_queue 3 config/llama3.1_dim_rc_hr_rfm/aim.yaml config/llama3.1_dim_rc_hr_rfm/cipher.yaml &
p3=$!
run_gpu_queue 4 config/llama3.1_dim_rc_hr_rfm/autodan.yaml config/llama3.1_dim_rc_hr_rfm/gcg.yaml &
p4=$!
run_gpu_queue 5 config/llama3.1_dim_rc_hr_rfm/pair.yaml config/llama3.1_dim_rc_hr_rfm/renellm.yaml &
p5=$!
run_gpu_queue 6 config/llama3.1_dim_rc_hr_rfm/xstest.yaml &
p6=$!

for p in $p1 $p2 $p3 $p4 $p5 $p6; do wait "$p"; done
echo "ALL_LLAMA_DIM_RC_HR_DONE"
