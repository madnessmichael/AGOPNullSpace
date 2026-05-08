#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

#python src/generate_response.py --config_path config/qwen2.5_rfm/aim.yaml
#python src/generate_response.py --config_path config/qwen2.5_rfm/alpaca_eval.yaml
#python src/generate_response.py --config_path config/qwen2.5_rfm/autodan.yaml
#python src/generate_response.py --config_path config/qwen2.5_rfm/cipher.yaml
#python src/generate_response.py --config_path config/qwen2.5_rfm/gcg.yaml
#python src/generate_response.py --config_path config/qwen2.5_rfm/gsm8k.yaml
#python src/generate_response.py --config_path config/qwen2.5_rfm/jailbroken.yaml
#python src/generate_response.py --config_path config/qwen2.5_rfm/pair.yaml
#python src/generate_response.py --config_path config/qwen2.5_rfm/renellm.yaml
#python src/generate_response.py --config_path config/qwen2.5_rfm/xstest.yaml
python src/generate_response.py --config_path config/qwen2.5_rfm/math.yaml
