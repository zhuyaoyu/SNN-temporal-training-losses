# Exploring Loss Functions for Time-based Training Strategy in Spiking Neural Networks

This repository is the official implementation of *Exploring Loss Functions for Time-based Training Strategy in Spiking Neural Networks* (**NeurIPS 2023**) \[[pdf](https://openreview.net/pdf?id=8IvW2k5VeA)\].

# Environment configuration
Regarding our code, we have implemented two backends for neuron functions in our algorithm: The python backend and the cuda backend, where the cuda backend significantly accelerates the neuron functions. The cuda backend requires additional environment configuration as shown in below.

The environment can be configured by the following steps:

1. Install pytorch (torchvision, torchaudio)

2. Install corresponding version of cuda-nvcc if you want to use the cuda backend (you may also use the default nvcc if the version is right, or you have admin account and install the right version):
	```shell
conda install -c nvidia cuda-nvcc
	```
	
3. Install other packages through requirements.txt:

   ```shell
   pip install -r requirements.txt
   ```

# Training

## Before running

Modify the data path and network settings in the .yaml config files (in the ./networks folder).

We recommend you to run the code in Linux environment, since we use pytorch cuda functions in the backward stage and the compile process is inconvenient in Windows environment.

The backend option can be configured by setting *backend: "cuda"* or *backend: "python"* in the .yaml config files.

## Run the code
```shell
CUDA_VISIBLE_DEVICES=0 python main.py -config networks/config_file.yaml
```
