#!/bin/sh

echo "BASE started =============================="
sudo apt update
sudo apt install python3-pip -y
sudo apt install mlocate -y
pip install gdown
pip install gradio
pip install wandb
echo "BASE ended ================================"

echo "CUDA started =============================="
sudo apt install software-properties-common -y
sudo apt install nvidia-cuda-toolkit -y
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
nvidia-smi
echo "CUDA ended ================================"

echo "LORA started =============================="
pip install -r ~/lora/requirements.txt
echo "LORA ended ================================"

echo "BASH started =============================="
echo 'export LD_LIBRARY_PATH=:/usr/lib/x86_64-linux-gnu/' >> ~/.bashrc
echo 'alias g3="python3"' >> ~/.bashrc
echo 'alias gl="git pull"' >> ~/.bashrc
. ~/.bashrc
echo "BASH ended ================================"

echo "ALL ended ================================="
