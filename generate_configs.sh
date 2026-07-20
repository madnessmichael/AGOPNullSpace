#!/bin/bash
# generate_configs.sh (fixed v2)
# Mỗi config chỉ 1 file cho mỗi (model, variant, dataset) — strength là 1 chuỗi
# liệt kê TOÀN BỘ các mức cần chạy, generate_response.py tự lặp qua trong 1 lần chạy.
set -euo pipefail

# ── Cấu hình — SỬA CHO ĐÚNG MÔI TRƯỜNG CỦA BẠN ──────────────────────────────
STRENGTHS="-7.0,-5.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,0.0,1.0,1.5,2.0,2.5,3.0,3.5,5.0,7.0"

# model:variant:filename  (giữ đúng tên file bạn đưa, kể cả chỗ "dim" không có "_rfm_")
MATRICES=(
    "gemma2:agopn:steering_matrix_gemma2_agopn_rfm_data1hh.pt"
    "gemma2:dim:steering_matrix_gemma2_dim_data1hh.pt"
    "llama3.1:agopn:steering_matrix_llama3.1_agopn_rfm_data1hh.pt"
    "llama3.1:dim:steering_matrix_llama3.1_dim_data1hh.pt"
    "qwen2.5:agopn:steering_matrix_qwen2.5_agopn_rfm_data1hh.pt"
    "qwen2.5:dim:steering_matrix_qwen2.5_dim_data1hh.pt"
)
STEERING_MATRIX_DIR="data/steering_matrix"

DATASETS=(aim autodan gcg cipher gsm8k pair renellm jailbroken xstest alpaca_eval math)

# Mỗi model cần có sẵn thư mục config gốc: config/<model>_rfm/<dataset>.yaml

NUM_GPUS=8
gpu_idx=0

echo "== Sinh config cho ${#MATRICES[@]} steering matrix × ${#DATASETS[@]} dataset =="
echo "== Mỗi config chạy toàn bộ dải strength: ${STRENGTHS}"

for entry in "${MATRICES[@]}"; do
    IFS=":" read -r model variant matrix_file <<< "$entry"

    SRC_DIR="config/${model}_rfm"
    if [[ ! -d "$SRC_DIR" ]]; then
        echo "!! Không tìm thấy thư mục config gốc: $SRC_DIR — bỏ qua model ${model}/${variant}"
        continue
    fi

    # Một thư mục DUY NHẤT cho mỗi model/variant, mỗi dataset chỉ 1 file
    dst_dir="config/${model}_${variant}_rfm"
    mkdir -p "$dst_dir"

    for ds in "${DATASETS[@]}"; do
        src="${SRC_DIR}/${ds}.yaml"
        dst="${dst_dir}/${ds}.yaml"
        if [[ ! -f "$src" ]]; then
            echo "!! Missing base config: $src — bỏ qua"
            continue
        fi
        cp "$src" "$dst"

        gpu=$(( gpu_idx % NUM_GPUS ))
        gpu_idx=$(( gpu_idx + 1 ))

        sed -i "s|^device:.*|device: cuda:${gpu}|" "$dst"
        sed -i "s|^steering_matrix_path:.*|steering_matrix_path: ${STEERING_MATRIX_DIR}/${matrix_file}|" "$dst"
        sed -i "s|^strength:.*|strength: \"${STRENGTHS}\"|" "$dst"
        # Chèn hậu tố duy nhất (model_variant) ngay trước ".json" trong output_file
        sed -i -E "/^output_file:/ s|\.json[[:space:]]*\$|_${model}_${variant}.json|" "$dst"

        echo "Đã sinh: ${dst} (GPU ${gpu})"
    done
done

# ── Kiểm tra trùng lặp output_file trên toàn bộ config vừa sinh ─────────────
echo ""
echo "Kiểm tra trùng lặp output_file trên toàn bộ config vừa sinh..."
dup=$(grep -h "^output_file:" config/*_rfm/*.yaml 2>/dev/null | sort | uniq -d || true)
if [[ -n "$dup" ]]; then
    echo "!!! PHÁT HIỆN OUTPUT_FILE TRÙNG — DỪNG LẠI, KHÔNG CHẠY GEN:"
    echo "$dup"
    exit 1
else
    echo "OK: tất cả output_file đều duy nhất."
fi