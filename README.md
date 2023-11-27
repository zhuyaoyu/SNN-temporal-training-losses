# Exploring Loss Functions for Time-based Training Strategy in Spiking Neural Networks

This repository is the official implementation of *Exploring Loss Functions for Time-based Training Strategy in Spiking Neural Networks* (**NeurIPS 2023**) \[[pdf](https://openreview.net/pdf?id=8IvW2k5VeA)\].

# Requirements
- pytorch=1.10.0
- torchvision=0.11.0
- spikingjelly

# Training

## Before running

Modify the data path and network settings in the .yaml config files (in the ./networks folder).

We recommend you to run the code in Linux environment, since we use pytorch cuda functions in the backward stage and the compile process is inconvenient in Windows environment.

In addition, we have implemented two backends for neuron functions in our algorithm: The python backend and the cuda backend, where the cuda backend significantly accelerates the neuron functions.

The backend option can be configured by setting **backend: "cuda"** or **backend: "python"** in the .yaml config files.

## Run the code
```
$ CUDA_VISIBLE_DEVICES=0 python main.py -config networks/config_file.yaml
```