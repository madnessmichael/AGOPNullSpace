


############### AlphaSteer ########################


# ── llama3.1-8b ─────────────────────────────────────────────────────────────────
python src/calc_steering_matrix.py \
    --model_name    llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --device        cuda \
    --save_path     data/steering_matrix/steering_matrix_llama3.1.pt


############### AGOP WITH NULLSPACE ########################


# ── llama3.3-70b ─────────────────────────────────────────────────────────────
python ./src/calc_steering_matrix_rfm.py \
    --model_name llama3.3-70b \
    --embedding_dir data/embeddings/llama3.3-70b \
    --device cuda \
    --save_path data/steering_matrix/steering_matrix_llama3.3-70b_rfm.pt \
    --rfm_method rfm \


# ── llama3.1-70b ─────────────────────────────────────────────────────────────
python ./src/calc_steering_matrix_rfm.py \
    --model_name llama3.1-70b \
    --embedding_dir data/embeddings/llama3.1-70b \
    --device cuda \
    --save_path data/steering_matrix/steering_matrix_llama3.1-70b_rfm.pt \
    --rfm_method rfm \

# ── llama3.1-8b ─────────────────────────────────────────────────────────────────
python ./src/calc_steering_matrix_rfm.py \
    --model_name llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --device cuda \
    --save_path data/steering_matrix/steering_matrix_llama3.1_rfm.pt \
    --rfm_method rfm



############### AGOP NO NULLSPACE ########################

# ── llama3.3-70b ─────────────────────────────────────────────────────────────
python src/calc_steering_matrix_rfm_no_nullspace.py \
    --model_name    llama3.3-70b \
    --embedding_dir data/embeddings/llama3.3-70b \
    --device        cuda \
    --save_path     data/steering_matrix/steering_matrix_llama3.3-70b_rfm_no_nullspace.pt \
    --rfm_method    rfm \


# ── llama3.1-70b ─────────────────────────────────────────────────────────────
python src/calc_steering_matrix_rfm_no_nullspace.py \
    --model_name    llama3.1-70b \
    --embedding_dir data/embeddings/llama3.1-70b \
    --device        cuda \
    --save_path     data/steering_matrix/steering_matrix_llama3.1-70b_rfm_no_nullspace.pt \
    --rfm_method    rfm \


# ── llama3.1-8b ─────────────────────────────────────────────────────────────────
python src/calc_steering_matrix_rfm_no_nullspace.py \
    --model_name    llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --device        cuda \
    --save_path     data/steering_matrix/steering_matrix_llama3.1_rfm_no_nullspace.pt \
    --rfm_method    rfm
    # llama3.1 extract không dùng --layers → không truyền vào đây




