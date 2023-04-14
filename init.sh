#!/bin/sh

echo "BASE started =============================="
sudo apt update
sudo apt install python3-pip -y
echo "BASE ended ================================"

echo "CUDA started =============================="
sudo apt install software-properties-common -y
sudo apt install nvidia-cuda-toolkit -y
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
invidia-smi
echo "CUDA ended ================================"

echo "LOAR started =============================="
git clone git@github.com:MJ2090/lora.git
pip install -r lora/requirements.txt
echo 'export LD_LIBRARY_PATH=:/usr/lib/x86_64-linux-gnu/' >> ~/.bashrc
source ~/.bashrc
echo "LOAR ended ================================"

echo "ALL ended ================================="
