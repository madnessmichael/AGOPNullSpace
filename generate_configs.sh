#!/bin/bash
# generate_configs.sh (fixed)
set -e

SRC_DIR="config/llama3.1_rfm"
MATRIX_IDS=("01" "03" "05" "07" "09")
declare -A GPU_MAP=( ["01"]=0 ["03"]=1 ["05"]=2 ["07"]=3 ["09"]=4 )

DATASETS=(aim autodan gcg cipher gsm8k pair renellm jailbroken xstest alpaca_eval math)

for mid in "${MATRIX_IDS[@]}"; do
    gpu="${GPU_MAP[$mid]}"
    dst_dir="config/llama3.1_rfm_${mid}"
    mkdir -p "$dst_dir"

    for ds in "${DATASETS[@]}"; do
        src="${SRC_DIR}/${ds}.yaml"
        dst="${dst_dir}/${ds}.yaml"

        if [[ ! -f "$src" ]]; then
            echo "!! Missing base config: $src — bỏ qua"
            continue
        fi

        cp "$src" "$dst"

        sed -i "s|^device:.*|device: cuda:${gpu}|" "$dst"
        sed -i "s|^steering_matrix_path:.*|steering_matrix_path: data/steering_matrix/steering_matrix_llama3.1_rfm_${mid}.pt|" "$dst"

        # FIX: match đúng đuôi thật của output_file, không phụ thuộc phần đầu path
        sed -i -E "s|_rfm_results\.json|_rfm_${mid}_results.json|" "$dst"

        sed -i "s|^strength:.*|strength: 0.75,|" "$dst"
    done
    echo "Đã sinh config cho matrix ${mid} (GPU ${gpu}) tại ${dst_dir}"
done

# ── BƯỚC KIỂM TRA BẮT BUỘC: đảm bảo không còn output_file trùng nhau ────────
echo ""
echo "Kiểm tra trùng lặp output_file trên toàn bộ config vừa sinh..."
dup=$(grep -h "^output_file:" config/llama3.1_rfm_*/*.yaml | sort | uniq -d)
if [[ -n "$dup" ]]; then
    echo "!!! PHÁT HIỆN OUTPUT_FILE TRÙNG — DỪNG LẠI, KHÔNG CHẠY GEN:"
    echo "$dup"
    exit 1
else
    echo "OK: tất cả output_file đều duy nhất."
fi