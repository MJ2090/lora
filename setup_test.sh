#!/bin/sh

echo "BASE started =============================="
pip install -U pip
pip install accelerate==0.18.0
pip install appdirs==1.4.4
pip install bitsandbytes==0.37.2
pip install datasets==2.10.1
pip install fire==0.5.0
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/huggingface/transformers.git
pip install torch==2.0.0
pip install sentencepiece==0.1.97
pip install tensorboardX==2.6
pip install gradio==3.23.0
echo "BASE started =============================="