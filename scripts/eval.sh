##python evaluation/jailbreak_llamaguard.py --input-dir data/responses/llama3.1/ --llamaguard-model meta-llama/Llama-Guard-3-8B --hf-token ***REMOVED***
##
##python evaluation/jailbreak_eval.py --input-file results.json
#
#
#for attack in aim autodan cipher gcg jailbroken pair renellm; do
#    python new_evaluation/jailbreak.py \
#        --input-file  data/responses/llama3.1/${attack}_llama3.1_results.json \
#        --output-file data/responses/llama3.1/${attack}_llama3.1_results_eval.json
#done
#
#
#python new_evaluation/xstest.py \
#    --input_file     data/responses/llama3.1/xstest_llama3.1_results.json \
#    --question_column prompt \
#    --model_name      llama3.1
#
#
## Xem kết quả tại strength tốt nhất (-0.5)
#python compute_results.py \
#    --responses_dir data/responses/llama3.1 \
#    --model_name    llama3.1 \
#    --strength      -0.5
#
## Hoặc xem tất cả 6 strengths cùng lúc
#python compute_results.py \
#    --responses_dir data/responses/llama3.1 \
#    --model_name    llama3.1 \
#    --all_strengths


#╔══════════════════════════════════════════════════════════════════════╗
#║                   Llama-Guard-3-8B as a judge                        ║
#╚══════════════════════════════════════════════════════════════════════╝

echo "=== Processing autodan ==="
python evaluation/jailbreak_llamaguard.py \
    --input-file data/responses/llama3.1/autodan_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-3-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing cipher ==="
python evaluation/jailbreak_llamaguard.py \
    --input-file data/responses/llama3.1/cipher_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-3-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing gcg ==="
python evaluation/jailbreak_llamaguard.py \
    --input-file data/responses/llama3.1/gcg_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-3-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing jailbroken ==="
python evaluation/jailbreak_llamaguard.py \
    --input-file data/responses/llama3.1/jailbroken_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-3-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing pair ==="
python evaluation/jailbreak_llamaguard.py \
    --input-file data/responses/llama3.1/pair_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-3-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing renellm ==="
python evaluation/jailbreak_llamaguard.py \
    --input-file data/responses/llama3.1/renellm_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-3-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1






#╔══════════════════════════════════════════════════════════════════════╗
#║                      Qwen3-32B-AWQ as a judge                        ║
#╚══════════════════════════════════════════════════════════════════════╝
echo "=== Processing autodan ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/autodan_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3-32B-AWQ \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing cipher ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/cipher_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3-32B-AWQ \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing gcg ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/gcg_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3-32B-AWQ \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing jailbroken ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/jailbroken_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3-32B-AWQ \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing pair ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/pair_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3-32B-AWQ \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing renellm ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/renellm_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3-32B-AWQ \
    --hf-token ***REMOVED*** \
    --batch-size 1



#╔══════════════════════════════════════════════════════════════════════╗
#║             Llama-3.3-70B-Instruct-bnb-4bit as a judge               ║
#╚══════════════════════════════════════════════════════════════════════╝

echo "=== Processing aim ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/aim_llama3.1_results.json \
    --llamaguard-model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing autodan ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/autodan_llama3.1_results.json \
    --llamaguard-model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing cipher ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/cipher_llama3.1_results.json \
    --llamaguard-model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing gcg ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/gcg_llama3.1_results.json \
    --llamaguard-model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing jailbroken ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/jailbroken_llama3.1_results.json \
    --llamaguard-model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing pair ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/pair_llama3.1_results.json \
    --llamaguard-model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing renellm ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/renellm_llama3.1_results.json \
    --llamaguard-model unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --hf-token ***REMOVED*** \
    --batch-size 1



#╔══════════════════════════════════════════════════════════════════════╗
#║                     Qwen3Guard-Gen-8B as a judge                     ║
#╚══════════════════════════════════════════════════════════════════════╝


echo "=== Processing aim ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/aim_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3Guard-Gen-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1


echo "=== Processing autodan ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/autodan_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3Guard-Gen-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing cipher ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/cipher_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3Guard-Gen-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing gcg ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/gcg_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3Guard-Gen-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing jailbroken ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/jailbroken_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3Guard-Gen-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing pair ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/pair_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3Guard-Gen-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing renellm ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/renellm_llama3.1_results.json \
    --llamaguard-model Qwen/Qwen3Guard-Gen-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1



#╔══════════════════════════════════════════════════════════════════════╗
#║                  Llama-Guard-4-12B as a judge                        ║
#╚══════════════════════════════════════════════════════════════════════╝

echo "=== Processing aim ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/aim_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing autodan ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/autodan_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing cipher ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/cipher_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing gcg ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/gcg_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing jailbroken ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/jailbroken_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing pair ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/pair_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing renellm ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/renellm_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1



#╔══════════════════════════════════════════════════════════════════════╗
#║                  Llama-Guard-4-12B as a judge--- RV                  ║
#╚══════════════════════════════════════════════════════════════════════╝



echo "=== Processing autodan ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/rv/autodan_llama3.1_rv_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing cipher ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/rv/cipher_llama3.1_rv_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing gcg ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/rv/gcg_llama3.1_rv_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing jailbroken ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/rv/jailbroken_llama3.1_rv_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing pair ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/rv/pair_llama3.1_rv_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing renellm ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/rv/renellm_llama3.1_rv_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1


#╔══════════════════════════════════════════════════════════════════════╗
#║                  Llama-Guard-4-12B as a judge RFM-AGOP               ║
#╚══════════════════════════════════════════════════════════════════════╝

echo "=== Processing aim ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/aim_llama3.1_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing autodan ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/autodan_llama3.1_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing cipher ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/cipher_llama3.1_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing gcg ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/gcg_llama3.1_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing jailbroken ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/jailbroken_llama3.1_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing pair ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/pair_llama3.1_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing renellm ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/renellm_llama3.1_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1


#╔══════════════════════════════════════════════════════════════════════╗
#║                qwen2.5  Llama-Guard-4-12B as a judge RFM-AGOP        ║
#╚══════════════════════════════════════════════════════════════════════╝

echo "=== Processing aim ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/aim_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing autodan ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/autodan_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing cipher ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/cipher_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing gcg ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/gcg_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing jailbroken ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/jailbroken_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing pair ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/pair_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing renellm ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/renellm_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1


#╔══════════════════════════════════════════════════════════════════════╗
#║                qwen2.5  Llama-Guard-4-12B as a judge RFM-AGOP        ║
#╚══════════════════════════════════════════════════════════════════════╝

echo "=== Processing aim ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/aim_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing autodan ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/autodan_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing cipher ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/cipher_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing gcg ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/gcg_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing jailbroken ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/jailbroken_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing pair ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/pair_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing renellm ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/qwen2.5/renellm_qwen2.5_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1


#╔══════════════════════════════════════════════════════════════════════╗
#║                gemma2  Llama-Guard-4-12B as a judge RFM-AGOP         ║
#╚══════════════════════════════════════════════════════════════════════╝

echo "=== Processing aim ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/gemma2/aim_gemma2_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing autodan ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/gemma2/autodan_gemma2_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing cipher ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/gemma2/cipher_gemma2_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing gcg ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/gemma2/gcg_gemma2_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing jailbroken ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/gemma2/jailbroken_gemma2_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing pair ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/gemma2/pair_gemma2_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1

echo "=== Processing renellm ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/gemma2/renellm_gemma2_rfm_results.json \
    --llamaguard-model meta-llama/Llama-Guard-4-12B \
    --hf-token ***REMOVED*** \
    --batch-size 1













echo "=== Processing autodan ==="
python evaluation/jailbreak_local.py \
    --input-file data/responses/llama3.1/autodan_llama3.1_results.json \
    --llamaguard-model meta-llama/Llama-Guard-3-8B \
    --hf-token ***REMOVED*** \
    --batch-size 1