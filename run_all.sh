#!/bin/bash
# run_all.sh (fixed v3) — chạy toàn bộ config đã sinh bởi generate_configs.sh
# Sửa lỗi: mỗi tiến trình con giờ được cô lập bằng CUDA_VISIBLE_DEVICES để
# không còn tự tạo "primary context" ~250MB thừa trên GPU 0.
set -uo pipefail
mkdir -p logs
mkdir -p /tmp/run_all_cfgs

DATASETS=(aim autodan gcg cipher gsm8k pair renellm jailbroken xstest alpaca_eval math)

NUM_PARALLEL="${NUM_PARALLEL:-1}"

# Đặt GPU_ID để chỉ chạy job được gán cho đúng GPU đó (dùng khi mở nhiều terminal song song).
# Ví dụ: GPU_ID=0 ./run_all.sh
GPU_ID="${GPU_ID:-}"

run_one() {
    local cfg="$1" tag="$2" ds="$3"

    # Lấy physical GPU id được gán trong config (dòng "device: cuda:N")
    local phys_gpu
    phys_gpu=$(grep -m1 -oP '^device:\s*cuda:\K[0-9]+' "$cfg" || echo "")

    if [[ -z "$phys_gpu" ]]; then
        echo "[${tag}] !! Không tìm thấy 'device: cuda:N' trong ${cfg} — bỏ qua"
        return
    fi

    # Tạo bản config tạm, đổi device thành cuda:0 (vì sau khi set CUDA_VISIBLE_DEVICES
    # thì trong mắt tiến trình con GPU được cấp phát luôn là index 0)
    local tmp_cfg="/tmp/run_all_cfgs/$(basename "$cfg" .yaml)_${tag}_$$.yaml"
    sed "s|^device:.*|device: cuda:0|" "$cfg" > "$tmp_cfg"

    echo "[${tag}] Bắt đầu ${ds} (GPU vật lý ${phys_gpu})"
    CUDA_VISIBLE_DEVICES="${phys_gpu}" python src/generate_response.py --config_path "$tmp_cfg" 2>&1 | tee -a "logs/${tag}_${ds}.log"
    echo "[${tag}] Xong ${ds}"

    rm -f "$tmp_cfg"
}
export -f run_one

jobs_file=$(mktemp)
for cfg_dir in config/*_rfm/; do
    model_variant=$(basename "$cfg_dir")
    for ds in "${DATASETS[@]}"; do
        cfg="${cfg_dir}${ds}.yaml"
        [[ -f "$cfg" ]] || continue
        if [[ -n "$GPU_ID" ]]; then
            grep -q "^device: *cuda:${GPU_ID}\$" "$cfg" || continue
        fi
        echo "${cfg}|${model_variant}|${ds}" >> "$jobs_file"
    done
done

echo "Tổng số job: $(wc -l < "$jobs_file")"

if [[ "$NUM_PARALLEL" -le 1 ]]; then
    while IFS="|" read -r cfg tag ds; do
        run_one "$cfg" "$tag" "$ds"
    done < "$jobs_file"
else
    xargs -a "$jobs_file" -d '\n' -P "$NUM_PARALLEL" -I{} bash -c '
        IFS="|" read -r cfg tag ds <<< "{}"
        run_one "$cfg" "$tag" "$ds"
    '
fi

rm -f "$jobs_file"