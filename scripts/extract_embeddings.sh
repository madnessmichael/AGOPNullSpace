huggingface-cli login --token ***REMOVED***


# #!/bin/bash

# TRAIN_VAL_DIR=data/instructions/train_val
# # Configuration - Change these variables to test different models and output directories
# EMBEDDING_DIR=data/embeddings/qwen2.5  # Output directory for embeddings
# NICKNAME=qwen2.5
# MODEL_NAME=Qwen/Qwen2.5-7B-Instruct  # Model name from HuggingFace

# DEVICE=cuda


# # Extract embeddings
# for file in $TRAIN_VAL_DIR/*.json; do
#    filename=$(basename "$file" .json)
#    echo "Extracting embeddings for $file"

#    # Set prompt_column based on filename
#    if [[ "$filename" == *"coconot"* ]]; then
#        prompt_column="prompt"
#    else
#        prompt_column="query"
#    fi

#    python src/extract_embeddings.py --model_name $MODEL_NAME \
#                                    --input_file $file \
#                                    --prompt_column "$prompt_column" \
#                                    --output_file $EMBEDDING_DIR/embeds_$filename.pt \
#                                    --batch_size 8 \
#                                    --device $DEVICE
# done

##!/bin/bash
#
#TRAIN_VAL_DIR=data/instructions/train_val
## Configuration - Change these variables to test different models and output directories
#EMBEDDING_DIR=data/embeddings/gemma2  # Output directory for embeddings
#NICKNAME=gemma2
#MODEL_NAME=google/gemma-2-9b-it  # Model name from HuggingFace
#
#DEVICE=cuda
#
#
## Extract embeddings
#for file in $TRAIN_VAL_DIR/*.json; do
#   filename=$(basename "$file" .json)
#   echo "Extracting embeddings for $file"
#
#   # Set prompt_column based on filename
#   if [[ "$filename" == *"coconot"* ]]; then
#       prompt_column="prompt"
#   else
#       prompt_column="query"
#   fi
#
#   python src/extract_embeddings.py --model_name $MODEL_NAME \
#                                   --input_file $file \
#                                   --prompt_column "$prompt_column" \
#                                   --output_file $EMBEDDING_DIR/embeds_$filename.pt \
#                                   --batch_size 1 \
#                                   --device $DEVICE
#done
#
#
##!/bin/bash
#
#TRAIN_VAL_DIR=data/instructions/train_val
## Configuration - Change these variables to test different models and output directories
#EMBEDDING_DIR=data/embeddings/llama3.1  # Output directory for embeddings
#NICKNAME=llama3.1
#MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct  # Model name from HuggingFace
#
#DEVICE=cuda:0
#
#
## Extract embeddings
#for file in $TRAIN_VAL_DIR/*.json; do
#    filename=$(basename "$file" .json)
#    echo "Extracting embeddings for $file"
#
#    # Set prompt_column based on filename
#    if [[ "$filename" == *"coconot"* ]]; then
#        prompt_column="prompt"
#    else
#        prompt_column="query"
#    fi
#
#    python src/extract_embeddings.py --model_name $MODEL_NAME \
#                                    --input_file $file \
#                                    --prompt_column "$prompt_column" \
#                                    --output_file $EMBEDDING_DIR/embeds_$filename.pt \
#                                    --batch_size 16 \
#                                    --device $DEVICE
#done
#
#
#
#


# #!/bin/bash

# TRAIN_VAL_DIR=data/instructions/train_val
# # Configuration - Change these variables to test different models and output directories
# EMBEDDING_DIR=data/embeddings/llama3.1-8b-unsloth  # Output directory for embeddings
# NICKNAME=llama3.1-8b-unsloth
# MODEL_NAME=unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit  # Model name from HuggingFace

# DEVICE=cuda


# # Extract embeddings
# for file in $TRAIN_VAL_DIR/*.json; do
#  filename=$(basename "$file" .json)
#  echo "Extracting embeddings for $file"

#  # Set prompt_column based on filename
#  if [[ "$filename" == *"coconot"* ]]; then
#      prompt_column="prompt"
#  else
#      prompt_column="query"
#  fi

#  python src/extract_embeddings.py --model_name $MODEL_NAME \
#                                  --input_file $file \
#                                  --prompt_column "$prompt_column" \
#                                  --output_file $EMBEDDING_DIR/embeds_$filename.pt \
#                                  --batch_size 64 \
#                                  --device $DEVICE
# done



#!/bin/bash

TRAIN_VAL_DIR=data/instructions/train_val
# Configuration - Change these variables to test different models and output directories
EMBEDDING_DIR=data/embeddings/llama3.3-70b  # Output directory for embeddings
NICKNAME=llama3.3-70b
MODEL_NAME=unsloth/Llama-3.3-70B-Instruct-bnb-4bit  # Model name from HuggingFace

DEVICE=cuda
LAYERS="28,30,32,34,36,38,40,42,44,46,48,50"



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
                                 --batch_size 2 \
                                 --layers $LAYERS \
                                 --device $DEVICE
done