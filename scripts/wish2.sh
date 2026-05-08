#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

#python src/generate_response.py --config_path config/gemma2_rfm/aim.yaml
#python src/generate_response.py --config_path config/gemma2_rfm/alpaca_eval.yaml
python src/generate_response.py --config_path config/gemma2_rfm/autodan.yaml
python src/generate_response.py --config_path config/gemma2_rfm/cipher.yaml
python src/generate_response.py --config_path config/gemma2_rfm/gcg.yaml
python src/generate_response.py --config_path config/gemma2_rfm/gsm8k.yaml
python src/generate_response.py --config_path config/gemma2_rfm/jailbroken.yaml
python src/generate_response.py --config_path config/gemma2_rfm/pair.yaml
python src/generate_response.py --config_path config/gemma2_rfm/renellm.yaml
python src/generate_response.py --config_path config/gemma2_rfm/xstest.yaml
python src/generate_response.py --config_path config/gemma2_rfm/math.yaml
