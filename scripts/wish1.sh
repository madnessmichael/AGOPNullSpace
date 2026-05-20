#!/bin/bash
export CUDA_VISIBLE_DEVICES=0







python src/generate_response.py --config_path config/llama3.3-70b_rfm/aim.yaml
python src/generate_response.py --config_path config/llama3.3-70b_rfm/cipher.yaml
python src/generate_response.py --config_path config/llama3.3-70b_rfm/gsm8k.yaml
python src/generate_response.py --config_path config/llama3.3-70b_rfm/pair.yaml
python src/generate_response.py --config_path config/llama3.3-70b_rfm/renellm.yaml
python src/generate_response.py --config_path config/llama3.3-70b_rfm/jailbroken.yaml
python src/generate_response.py --config_path config/llama3.3-70b_rfm/xstest.yaml
python src/generate_response.py --config_path config/llama3.3-70b_rfm/alpaca_eval.yaml\
python src/generate_response.py --config_path config/llama3.3-70b_rfm/math.yaml
