#!/bin/bash
# run_extract_rc_style_full.sh
# Extracts activations for ALL 9236 rows of refusal_compliance_full_hard_refusal
# WITH prompt_style/category/label metadata preserved per row (needed for the
# DIM-vs-RFM per-style-family AUC analysis; the production embeds_refusal.pt/
# embeds_compliance.pt files don't preserve this mapping -- see
# evaluation/extract_rc_style_sample.py docstring).
#
# Each style runs as its OWN subprocess (--only_style), giving it a fresh CUDA
# context -- two earlier attempts running all 21 styles in one long-lived
# process both OOM'd from allocator fragmentation building up across styles
# (even with expandable_segments + empty_cache between styles), despite no
# single style being individually memory-heavy at these batch sizes.
set -uo pipefail
cd /home/workspace/mad_workspace/LLM/agopns_clean/AGOPNullSpace
mkdir -p logs/extract_rc_style/styles

RC_JSON=data/embeddings/llama3.1/refusal_compliance_full_hard_refusal/sorrybench_rc_dataset.json
OUT_DIR=data/embeddings/llama3.1/rc_style_full_dir
MODEL=meta-llama/Llama-3.1-8B-Instruct

run_gpu_queue() {
    local gpu="$1"; shift
    for style in "$@"; do
        logfile="logs/extract_rc_style/styles/${style}.log"
        echo "[$(date +%H:%M:%S)] [gpu${gpu}] extracting ${style} -> ${logfile}"
        if python3 evaluation/extract_rc_style_sample.py \
            --model_name "$MODEL" --rc_json "$RC_JSON" --out_file "$OUT_DIR" \
            --n_per_style 500 --only_style "$style" --device "cuda:${gpu}" \
            > "$logfile" 2>&1; then
            echo "[$(date +%H:%M:%S)] [gpu${gpu}] finished ${style}"
        else
            echo "[$(date +%H:%M:%S)] [gpu${gpu}] !!! FAILED ${style} -- see ${logfile} !!!"
        fi
    done
}

run_gpu_queue 0 ascii atbash authority_endorsement &
p0=$!
run_gpu_queue 1 base caesar evidence-based_persuasion &
p1=$!
run_gpu_queue 2 expert_endorsement logical_appeal misrepresentation &
p2=$!
run_gpu_queue 3 misspellings morse question &
p3=$!
run_gpu_queue 4 role_play slang technical_terms &
p4=$!
run_gpu_queue 5 translate-fr translate-ml translate-mr &
p5=$!
run_gpu_queue 6 translate-ta translate-zh-cn uncommon_dialects &
p6=$!

for p in $p0 $p1 $p2 $p3 $p4 $p5 $p6; do wait "$p"; done
echo "ALL_RC_STYLE_FULL_EXTRACT_DONE"
