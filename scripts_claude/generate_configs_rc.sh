#!/bin/bash
# generate_configs_rc.sh
# Config generator for the SORRY-Bench refuse-compliance (rc) sweep.
# Adapted from generate_configs.sh: same copy+sed pattern, but:
#   - base configs come from <model>_agopn_rfm (already has per-dataset
#     batch_size/max_new_tokens/prompt_column tuned correctly)
#   - two variants per model: rc_ns (nullspace+gate, steering_matrix_path)
#     and rc_nons (pure additive, steering_vector_path -- note the KEY NAME
#     changes, not just the value, since generate_response.py branches on
#     which key is present)
#   - strength is the new sweep {0.0,...,1.5}, and alpaca_eval is dropped
#     from DATASETS (7 attacks + 3 utility only)
set -euo pipefail

STRENGTHS="0.0,0.3,0.5,0.7,0.9,1.0,1.3,1.5"
STEERING_MATRIX_DIR="data/steering_matrix"

# model:variant:key:filename
#   Both variants now save the rank1_gate_v1 dict format and load via
#   steering_matrix_path (AlphaSteer_MODELS_DICT) -- rc_ns has real u+gate
#   per layer, rc_nons has u=None per layer (no-gate, pure additive branch).
#   NaiveSteerModel/steering_vector_path is no longer used by this pipeline.
ENTRIES=(
    "llama3.1:rc_ns:steering_matrix_path:steering_matrix_llama3.1_rfm_rc.pt"
    "llama3.1:rc_nons:steering_matrix_path:steering_matrix_llama3.1_rfm_rc_no_nullspace.pt"
    "qwen2.5:rc_ns:steering_matrix_path:steering_matrix_qwen2.5_rfm_rc.pt"
    "qwen2.5:rc_nons:steering_matrix_path:steering_matrix_qwen2.5_rfm_rc_no_nullspace.pt"
    "gemma2:rc_ns:steering_matrix_path:steering_matrix_gemma2_rfm_rc.pt"
    "gemma2:rc_nons:steering_matrix_path:steering_matrix_gemma2_rfm_rc_no_nullspace.pt"
)

# 7 attacks + 3 utility -- alpaca_eval excluded
DATASETS=(aim autodan cipher gcg jailbroken pair renellm gsm8k math xstest)

NUM_GPUS=8
gpu_idx=0

echo "== Generating config for ${#ENTRIES[@]} model/variant x ${#DATASETS[@]} dataset =="
echo "== strength sweep: ${STRENGTHS}"

for entry in "${ENTRIES[@]}"; do
    IFS=":" read -r model variant key matrix_file <<< "$entry"

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

        # Base yaml files have NO trailing newline -- appending with plain
        # `echo >>` concatenates onto the last line instead of starting a
        # new one, and the strength sed below (greedy `.*`) then silently
        # swallows the appended key entirely. Force a real newline first.
        printf '\n' >> "$dst"

        sed -i "s|^device:.*|device: cuda:${gpu}|" "$dst"
        # Drop whichever path key the base config shipped with, then insert
        # the correct key for this variant (steering_matrix_path vs
        # steering_vector_path -- generate_response.py branches on which
        # attribute is present, so both must never coexist in one file).
        sed -i "/^steering_matrix_path:/d; /^steering_vector_path:/d" "$dst"
        printf '%s: %s/%s\n' "${key}" "${STEERING_MATRIX_DIR}" "${matrix_file}" >> "$dst"
        sed -i "s|^strength:.*|strength: \"${STRENGTHS}\"|" "$dst"
        sed -i -E "/^output_file:/ s|\.json[[:space:]]*\$|_${model}_${variant}.json|" "$dst"

        echo "Generated: ${dst} (GPU ${gpu})"
    done
done

echo ""
echo "Checking output_file collisions across all generated configs..."
dup=$(grep -h "^output_file:" config/*_rc_*_rfm/*.yaml 2>/dev/null | sort | uniq -d || true)
if [[ -n "$dup" ]]; then
    echo "!!! DUPLICATE output_file FOUND -- STOPPING, DO NOT RUN GENERATION:"
    echo "$dup"
    exit 1
else
    echo "OK: all output_file values are unique."
fi
