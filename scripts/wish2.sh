#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python src/generate_response.py --config_path config/gemma2_rfm/aim.yaml
python src/generate_response.py --config_path config/gemma2_rfm/alpaca_eval.yaml  
CUDA_VISIBLE_DEVICES=2 python src/generate_response.py --config_path config/gemma2_rfm/autodan.yaml
CUDA_VISIBLE_DEVICES=3 python src/generate_response.py --config_path config/gemma2_rfm/cipher.yaml
CUDA_VISIBLE_DEVICES=4 python src/generate_response.py --config_path config/gemma2_rfm/gcg.yaml
python src/generate_response.py --config_path config/gemma2_rfm/gsm8k.yaml 
CUDA_VISIBLE_DEVICES=5 python src/generate_response.py --config_path config/gemma2_rfm/jailbroken.yaml
CUDA_VISIBLE_DEVICES=7 python src/generate_response.py --config_path config/gemma2_rfm/pair.yaml
python src/generate_response.py --config_path config/gemma2_rfm/renellm.yaml 
python src/generate_response.py --config_path config/gemma2_rfm/xstest.yaml 
python src/generate_response.py --config_path config/gemma2_rfm/math.yaml 
