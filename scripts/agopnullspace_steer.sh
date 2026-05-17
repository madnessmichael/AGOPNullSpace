


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
    --layers 28,30,32,34,36,38,40,42,44,46,48,50


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
    --layers        28,30,32,34,36,38,40,42,44,46,48,50

# ── llama3.1 ─────────────────────────────────────────────────────────────────
python src/calc_steering_matrix_rfm_no_nullspace.py \
    --model_name    llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --device        cuda \
    --save_path     data/steering_matrix/steering_matrix_llama3.1_rfm_no_nullspace.pt \
    --rfm_method    rfm
    # llama3.1 extract không dùng --layers → không truyền vào đây