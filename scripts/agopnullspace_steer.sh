


############### AlphaSteer ########################


# llama3.3-70b — dùng --layers vì embed với --layers
python src/calc_steering_matrix.py \
    --model_name    llama3.3-70b \
    --embedding_dir data/embeddings/llama3.3-70b \
    --device        cuda \
    --save_path     data/steering_matrix/steering_matrix_llama3.3-70b.pt \
    --layers        28,30,32,34,36,38,40,42,44,46,48,50

# llama3.1 — không dùng --layers vì embed toàn bộ layers
python src/calc_steering_matrix.py \
    --model_name    llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --device        cuda \
    --save_path     data/steering_matrix/steering_matrix_llama3.1.pt


############### AGOP WITH NULLSPACE ########################



python ./src/calc_steering_matrix_rfm.py \
    --model_name llama3.3-70b \
    --embedding_dir data/embeddings/llama3.3-70b \
    --device cuda \
    --save_path data/steering_matrix/steering_matrix_llama3.3-70b_rfm.pt \
    --rfm_method rfm \



GENERATE_CONFIG_DIR=config/llama3.3-70b_rfm && echo "Generating response for $NICKNAME" && for file in $GENERATE_CONFIG_DIR/*.yaml; do filename=$(basename "$file" .yaml); echo "Generating response for $file"; python src/generate_response.py --config_path "$file"; done


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



############### AGOP NO NULLSPACE ########################

# ── llama3.3-70b ─────────────────────────────────────────────────────────────
python src/calc_steering_matrix_rfm_no_nullspace.py \
    --model_name    llama3.3-70b \
    --embedding_dir data/embeddings/llama3.3-70b \
    --device        cuda \
    --save_path     data/steering_matrix/steering_matrix_llama3.3-70b_rfm_no_nullspace.pt \
    --rfm_method    rfm \


# Steering!
GENERATE_CONFIG_DIR=config/llama3.3-70b_rfm_no_nullspace
echo "Generating response for $NICKNAME"
for file in $GENERATE_CONFIG_DIR/*.yaml; do
    filename=$(basename "$file" .yaml)
    echo "Generating response for $file"
    python src/generate_response.py --config_path $file
done


# ── llama3.1 ─────────────────────────────────────────────────────────────────
python src/calc_steering_matrix_rfm_no_nullspace.py \
    --model_name    llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --device        cuda \
    --save_path     data/steering_matrix/steering_matrix_llama3.1_rfm_no_nullspace.pt \
    --rfm_method    rfm
    # llama3.1 extract không dùng --layers → không truyền vào đây

GENERATE_CONFIG_DIR=config/llama3.1_rfm_no_nullspace
echo "Generating response for $NICKNAME"
for file in $GENERATE_CONFIG_DIR/*.yaml; do
    filename=$(basename "$file" .yaml)
    echo "Generating response for $file"
    python src/generate_response.py --config_path $file
done







# ── rfm_refusal_vector_v4 ────BALANCING─────────────────────────────────────────────────────────────


python ./src/rfm_refusal_vector_v4.py --embedding_dir ./data/embeddings/gemma2 --model_name gemma2 --method rfm --device cuda --save_path ./data/refusal_vectors/RFM_v4/gemma2_rfm_v4.pkl

python ./src/rfm_refusal_vector_v4.py --embedding_dir ./data/embeddings/llama3.1 --model_name llama3.1 --method rfm --device cuda --save_path ./data/refusal_vectors/RFM_v4/llama3.1_rfm_v4.pkl

python ./src/rfm_refusal_vector_v4.py --embedding_dir ./data/embeddings/qwen2.5 --model_name qwen2.5 --method rfm --device cuda --save_path ./data/refusal_vectors/RFM_v4/qwen2.5_rfm_v4.pkl



# ── rfm_refusal_vector_v5 ────BALANCING────────harmful_harmless_instruction──────────────────────────


python ./src/prepare_harmful_harmless_instruction_embeddings.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_dir ./data/embeddings/llama3.1/harmful_harmless_instructions \
  --split train \
  --batch_size 2 \
  --device cuda \
  --dtype bfloat16 \
  --save_separated


python ./src/prepare_harmful_harmless_instruction_embeddings.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_dir ./data/embeddings/llama3.1/harmful_harmless_instructions \
  --split train \
  --batch_size 2 \
  --device cuda \
  --dtype bfloat16 \
  --save_separated


python ./src/prepare_harmful_harmless_instruction_embeddings.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_dir ./data/embeddings/llama3.1/harmful_harmless_instructions \
  --split train \
  --batch_size 2 \
  --device cuda \
  --dtype bfloat16 \
  --save_separated



python ./src/rfm_refusal_vector_hh_v5.py \
  --model_name llama3.1 \
  --embedding_dir ./data/embeddings/llama3.1/harmful_harmless_instructions \
  --method rfm \
  --rfm_iters 3 \
  --device cuda \
  --balance_ratio 1.0 \
  --save_path data/refusal_vectors/refusal_vector_llama3.1_v5hh_rfm.pkl


python ./src/calc_steering_matrix_rfm_hh_v5.py \
  --model_name llama3.1 \
  --embedding_dir ./data/embeddings/llama3.1/harmful_harmless_instructions \
  --device cuda \
  --save_path data/steering_matrix/steering_matrix_llama3.1_v5hh_rfm.pt \
  --rfm_method rfm \
  --rfm_iters 3 \
  --balance_ratio 1.0


# ──closed─form────AGOP────Constrained Rayleigh-Ritz──────────────────────────


python ./src/rfm_refusal_vector_cr_steer.py \
  --embedding_dir ./data/embeddings/qwen2.5 \
  --model_name qwen2.5 \
  --dim_pkl_path ./data/refusal_vectors/RV/qwen2.5_RV_refusal.pkl \
  --method rfm \
  --save_path ./data/refusal_vectors/CRSTEER/qwen2.5_CRSTEER_refusal.pkl


python ./src/calc_steering_matrix_rfm.py \
    --embedding_dir ./data/embeddings/qwen2.5 \
    --model_name qwen2.5 \
    --refusal_vector_path ./data/refusal_vectors/CRSTEER/qwen2.5_CRSTEER_refusal.pkl \
    --lambda_reg 4.0 \
    --save_path ./data/steering_matrices/qwen2.5_CRSTEER_matrix.pt