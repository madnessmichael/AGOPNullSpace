#!/bin/bash
# scripts/rv.sh

mkdir -p data/responses/llama3.1/rv

#ATTACKS="autodan cipher gcg jailbroken pair renellm"
ATTACKS="alpaca_eval xstest gsm8k math"


for ATTACK in $ATTACKS; do
    CONFIG="config/llama3.1/${ATTACK}.yaml"

    # Tạo config tạm bằng cách đọc file gốc, override các field cần thiết
    python - <<EOF
import yaml

with open("${CONFIG}") as f:
    cfg = yaml.safe_load(f)

# Override chỉ những field khác
cfg.pop("steering_matrix_path", None)
cfg["steering_vector_path"] = "data/refusal_vectors/RV/llama3.1_RV_refusal.pkl"
cfg["output_file"] = cfg["output_file"].replace("llama3.1_results", "llama3.1_rv_results").replace(
    "data/responses/llama3.1/", "data/responses/llama3.1/rv/")
cfg["strength"] = "0.0,0.1,0.2,0.3,0.4,0.5"
cfg["batch_size"] = 4  # RV dùng batch nhỏ hơn

with open("./config/llama3.1_rv/rv_${ATTACK}.yaml", "w") as f:
    yaml.dump(cfg, f)

print("Config for ${ATTACK}:")
print(yaml.dump(cfg))
EOF

    echo "=== Processing ${ATTACK} ==="
    python src/generate_response_rv.py --config_path ./config/llama3.1_rv/rv_${ATTACK}.yaml
done