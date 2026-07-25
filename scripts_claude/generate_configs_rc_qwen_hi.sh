#!/bin/bash
# generate_configs_rc_qwen_hi.sh
# One-off follow-up: qwen2.5 only, both variants (rc_ns / rc_nons), high-strength
# sweep {8.0,8.5,9.0,9.5,10.0,10.5,11.0} across all 10 datasets (7 attack + 3
# utility, alpaca_eval excluded -- same scope as the main _rc sweep). Separate
# config dirs / output_file suffix ("_hi") so results don't collide with the
# existing 0.0-1.5 sweep.
set -euo pipefail

STRENGTHS="8.0,8.5,9.0,9.5,10.0,10.5,11.0"
STEERING_MATRIX_DIR="data/steering_matrix"

ENTRIES=(
    "qwen2.5:rc_ns_hi:steering_matrix_path:steering_matrix_qwen2.5_rfm_rc.pt"
    "qwen2.5:rc_nons_hi:steering_matrix_path:steering_matrix_qwen2.5_rfm_rc_no_nullspace.pt"
)

DATASETS=(aim autodan cipher gcg jailbroken pair renellm gsm8k math xstest)

NUM_GPUS=7  # 0-6 only -- GPU 7 reserved for interactive use (inference_sample.ipynb)
gpu_idx=0

echo "== Generating ${#ENTRIES[@]} variant x ${#DATASETS[@]} dataset (qwen2.5 high-strength) =="
echo "== strength sweep: ${STRENGTHS}"

for entry in "${ENTRIES[@]}"; do
    IFS=":" read -r model variant key matrix_file <<< "$entry"
    SRC_DIR="config/${model}_agopn_rfm"
    dst_dir="config/${model}_${variant}_rfm"
    mkdir -p "$dst_dir"

    for ds in "${DATASETS[@]}"; do
        src="${SRC_DIR}/${ds}.yaml"
        dst="${dst_dir}/${ds}.yaml"
        [[ -f "$src" ]] || { echo "!! Missing base config: $src -- skipping"; continue; }
        cp "$src" "$dst"

        gpu=$(( gpu_idx % NUM_GPUS ))
        gpu_idx=$(( gpu_idx + 1 ))

        printf '\n' >> "$dst"   # base yaml has no trailing newline -- see generate_configs_rc.sh fix
        sed -i "s|^device:.*|device: cuda:${gpu}|" "$dst"
        sed -i "/^steering_matrix_path:/d; /^steering_vector_path:/d" "$dst"
        printf '%s: %s/%s\n' "${key}" "${STEERING_MATRIX_DIR}" "${matrix_file}" >> "$dst"
        sed -i "s|^strength:.*|strength: \"${STRENGTHS}\"|" "$dst"
        sed -i -E "/^output_file:/ s|\.json[[:space:]]*\$|_${model}_${variant}.json|" "$dst"

        echo "Generated: ${dst} (GPU ${gpu})"
    done
done

echo ""
echo "Checking output_file collisions..."
dup=$(grep -h "^output_file:" config/qwen2.5_rc_ns_hi_rfm/*.yaml config/qwen2.5_rc_nons_hi_rfm/*.yaml 2>/dev/null | sort | uniq -d || true)
if [[ -n "$dup" ]]; then
    echo "!!! DUPLICATE output_file -- STOPPING:"
    echo "$dup"
    exit 1
fi
# Also check against the EXISTING 0.0-1.5 sweep's output files -- must not collide.
dup2=$(grep -h "^output_file:" config/qwen2.5_rc_ns_hi_rfm/*.yaml config/qwen2.5_rc_nons_hi_rfm/*.yaml config/qwen2.5_rc_ns_rfm/*.yaml config/qwen2.5_rc_nons_rfm/*.yaml 2>/dev/null | sort | uniq -d || true)
if [[ -n "$dup2" ]]; then
    echo "!!! COLLISION WITH EXISTING 0.0-1.5 SWEEP output_file -- STOPPING:"
    echo "$dup2"
    exit 1
fi
echo "OK: all output_file values unique (including vs. the existing 0.0-1.5 sweep)."
