# VA-eval
Code &amp; datasets for my thesis about voice assistant evaluation
This repository contains the custom scripts, dataset samples, and configurations 
used for my thesis experiments based on the ParlAI framework.

The ParlAI framework itself is not included here.  
Official ParlAI repository: https://github.com/facebookresearch/ParlAI

The dataset I used for evaluation: https://huggingface.co/datasets/AudioLLMs/alpaca_audio_test
The model I used as baseline: blender_90M

Latency Evaluation Code have 3 parts:
1. blender_latency.py
2. asr_latency.py
3. tts_latency.py 

For scripts：
check_cuda.py
convert_csv_to_parlai_eval.py.py
generate_mydata.py
These scripts extend the functionality of the official ParlAI framework.
They do not modify ParlAI core files.
