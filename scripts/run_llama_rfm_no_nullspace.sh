#!/bin/bash

TRAIN_VAL_DIR=data/instructions/train_val
EMBEDDING_DIR=data/embeddings/llama3.1
NICKNAME=llama3.1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
DEVICE=cuda

# Extract embeddings (giữ nguyên)
for file in $TRAIN_VAL_DIR/*.json; do
    filename=$(basename "$file" .json)
    echo "Extracting embeddings for $file"

    if [[ "$filename" == *"coconot"* ]]; then
        prompt_column="prompt"
    else
        prompt_column="query"
    fi

    python src/extract_embeddings.py --model_name $MODEL_NAME \
                                    --input_file $file \
                                    --prompt_column "$prompt_column" \
                                    --output_file $EMBEDDING_DIR/embeds_$filename.pt \
                                    --batch_size 16 \
                                    --device $DEVICE
done

# Tính refusal vector bằng RFM (giữ nguyên)
python ./src/rfm_refusal_vector.py \
    --embedding_dir data/embeddings/llama3.1 \
    --model_name llama3.1 \
    --method rfm \
    --device cuda \
    --save_path data/refusal_vectors/RFM/llama3.1_RFM_refusal.pkl

# *** THAY ĐỔI: dùng script không NullSpace ***
python ./src/calc_steering_matrix_rfm_no_nullspace.py \
    --model_name llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --device cuda \
    --save_path data/steering_matrix/steering_matrix_llama3.1_rfm_no_nullspace.pt \
    --rfm_method rfm

# Steering — trỏ config mới
GENERATE_CONFIG_DIR=config/llama3.1_rfm_no_nullspace
echo "Generating response for $NICKNAME (no nullspace)"
for file in $GENERATE_CONFIG_DIR/*.yaml; do
    filename=$(basename "$file" .yaml)
    echo "Generating response for $file"
    python src/generate_response.py --config_path $file
done