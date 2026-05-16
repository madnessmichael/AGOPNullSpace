
#!/bin/bash

TRAIN_VAL_DIR=data/instructions/train_val
# Configuration - Change these variables to test different models and output directories
EMBEDDING_DIR=data/embeddings/llama3.1  # Output directory for embeddings
NICKNAME=llama3.1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct  # Model name from HuggingFace

DEVICE=cuda



# Extract embeddings
for file in $TRAIN_VAL_DIR/*.json; do
    filename=$(basename "$file" .json)
    echo "Extracting embeddings for $file"

    # Set prompt_column based on filename
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


python ./src/rfm_refusal_vector.py \
    --embedding_dir data/embeddings/llama3.1 \
    --model_name llama3.1 \
    --method rfm \
    --device cuda \
    --save_path data/refusal_vectors/RFM/llama3.1_RFM_refusal.pkl


python ./src/calc_steering_matrix_rfm.py \
    --model_name llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --device cuda \
    --save_path data/steering_matrix/steering_matrix_llama3.1_rfm.pt \
    --rfm_method rfm


# Steering!
GENERATE_CONFIG_DIR=config/llama3.1_rfm
echo "Generating response for $NICKNAME"
for file in $GENERATE_CONFIG_DIR/*.yaml; do
    filename=$(basename "$file" .yaml)
    echo "Generating response for $file"
    python src/generate_response.py --config_path $file
done











#
#python ./src/rfm_refusal_vector.py \
#    --embedding_dir data/embeddings/qwen2.5 \
#    --model_name qwen2.5 \
#    --method rfm \
#    --device cuda \
#    --save_path data/refusal_vectors/RFM/qwen2.5_RFM_refusal.pkl
#
#
#python ./src/calc_steering_matrix_rfm.py \
#    --model_name qwen2.5 \
#    --embedding_dir data/embeddings/qwen2.5 \
#    --device cuda \
#    --save_path data/steering_matrix/steering_matrix_qwen2.5_rfm.pt \
#    --rfm_method rfm
#
#
### Configuration - Change these variables to test different models and output directories
##NICKNAME=qwen2.5
##DEVICE=cuda
### Steering!
##GENERATE_CONFIG_DIR=config/qwen2.5_rfm
##echo "Generating response for $NICKNAME"
##for file in $GENERATE_CONFIG_DIR/*.yaml; do
##    filename=$(basename "$file" .yaml)
##    echo "Generating response for $file"
##    python src/generate_response.py --config_path $file
##done
#
#
#
#





python ./src/rfm_refusal_vector.py \
    --embedding_dir data/embeddings/gemma2 \
    --model_name gemma2 \
    --method rfm \
    --device cuda \
    --save_path data/refusal_vectors/RFM/gemma2_RFM_refusal.pkl


python ./src/calc_steering_matrix_rfm.py \
    --model_name gemma2 \
    --embedding_dir data/embeddings/gemma2 \
    --device cuda \
    --save_path data/steering_matrix/steering_matrix_gemma2_rfm.pt \
    --rfm_method rfm


## Configuration - Change these variables to test different models and output directories
#NICKNAME=gemma2
#DEVICE=cuda
## Steering!
#GENERATE_CONFIG_DIR=config/gemma2_rfm
#echo "Generating response for $NICKNAME"
#for file in $GENERATE_CONFIG_DIR/*.yaml; do
#    filename=$(basename "$file" .yaml)
#    echo "Generating response for $file"
#    python src/generate_response.py --config_path $file
#done







