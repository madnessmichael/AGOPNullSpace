#!/bin/bash
# generate_configs_rc_full.sh
# Config generator for the refusal_compliance_full-trained steering matrices
# (9,236-row SORRY-Bench dataset), combined single sweep over BOTH the original
# 0.0-1.5 range and the qwen high-strength 8.0-11.0 range in one run:
#   {0.0,0.3,0.5,0.7,0.9,1.0,1.3,1.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0}
# Two variants per model (rc_ns_full = nullspace+gate, rc_nons_full = no-gate
# additive), matrices are steering_matrix_<model>_rfm_rc_full[.pt|_no_nullspace.pt].
# Output filenames get an "_full" suffix so they never collide with the
# existing rc_ns/rc_nons (440-prompt-trained) sweep results.
set -euo pipefail

STRENGTHS="0.0,0.3,0.5,0.7,0.9,1.0,1.3,1.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0"
STEERING_MATRIX_DIR="data/steering_matrix"

# model:variant:key:filename
ENTRIES=(
    "qwen2.5:rc_ns_full:steering_matrix_path:steering_matrix_qwen2.5_rfm_rc_full.pt"
    "qwen2.5:rc_nons_full:steering_matrix_path:steering_matrix_qwen2.5_rfm_rc_full_no_nullspace.pt"
    "llama3.1:rc_ns_full:steering_matrix_path:steering_matrix_llama3.1_rfm_rc_full.pt"
    "llama3.1:rc_nons_full:steering_matrix_path:steering_matrix_llama3.1_rfm_rc_full_no_nullspace.pt"
    "gemma2:rc_ns_full:steering_matrix_path:steering_matrix_gemma2_rfm_rc_full.pt"
    "gemma2:rc_nons_full:steering_matrix_path:steering_matrix_gemma2_rfm_rc_full_no_nullspace.pt"
)

DATASETS=(aim autodan cipher gcg jailbroken pair renellm gsm8k math xstest)
NUM_GPUS=8

echo "== Generating configs: ${#ENTRIES[@]} model/variant x ${#DATASETS[@]} dataset =="
echo "== strength sweep: ${STRENGTHS}"

prev_model=""
for entry in "${ENTRIES[@]}"; do
    IFS=":" read -r model variant key matrix_file <<< "$entry"

    # Reset GPU round-robin at the start of each model so its own ~20 jobs
    # (2 variants x 10 datasets) spread evenly across all 8 GPUs.
    if [[ "$model" != "$prev_model" ]]; then
        gpu_idx=0
        prev_model="$model"
    fi

    SRC_DIR="config/${model}_agopn_rfm"
    if [[ ! -d "$SRC_DIR" ]]; then
        echo "!! Missing base config dir: $SRC_DIR -- skipping ${model}/${variant}"
        continue
    fi

    dst_dir="config/${model}_${variant}_rfm"
    mkdir -p "$dst_dir"

    for ds in "${DATASETS[@]}"; do
        src="${SRC_DIR}/${ds}.yaml"
        dst="${dst_dir}/${ds}.yaml"
        if [[ ! -f "$src" ]]; then
            echo "!! Missing base config: $src -- skipping"
            continue
        fi
        cp "$src" "$dst"

        gpu=$(( gpu_idx % NUM_GPUS ))
        gpu_idx=$(( gpu_idx + 1 ))

        printf '\n' >> "$dst"   # base yaml has no trailing newline -- see generate_configs_rc.sh
        sed -i "s|^device:.*|device: cuda:${gpu}|" "$dst"
        sed -i "/^steering_matrix_path:/d; /^steering_vector_path:/d" "$dst"
        printf '%s: %s/%s\n' "${key}" "${STEERING_MATRIX_DIR}" "${matrix_file}" >> "$dst"
        sed -i "s|^strength:.*|strength: \"${STRENGTHS}\"|" "$dst"
        sed -i -E "/^output_file:/ s|\.json[[:space:]]*\$|_${model}_${variant}.json|" "$dst"

        echo "Generated: ${dst} (GPU ${gpu})"
    done
done

echo ""
echo "Checking output_file collisions across all generated _full configs..."
dup=$(grep -h "^output_file:" config/*_rc_*_full_rfm/*.yaml 2>/dev/null | sort | uniq -d || true)
if [[ -n "$dup" ]]; then
    echo "!!! DUPLICATE output_file FOUND -- STOPPING, DO NOT RUN GENERATION:"
    echo "$dup"
    exit 1
fi
echo "OK: all _full output_file values unique."

echo ""
echo "Checking against ALL existing output_file values (old sweeps) for collisions..."
all_yaml=$(ls config/*_rc_*_rfm/*.yaml 2>/dev/null | sort -u)
dup2=$(grep -h "^output_file:" $all_yaml 2>/dev/null | sort | uniq -d || true)
if [[ -n "$dup2" ]]; then
    echo "!!! COLLISION WITH EXISTING SWEEP output_file -- STOPPING:"
    echo "$dup2"
    exit 1
fi
echo "OK: no collision with existing sweeps."
