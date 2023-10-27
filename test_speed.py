import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from spikingjelly.activation_based import cuda_utils, functional, neuron, surrogate
import argparse
from typing import Callable
import global_v as glv
from network_parser import parse
import cnns
import layers.losses as losses
from main import get_loss


def cuda_timer(device: torch.device or int, f: Callable, *args, **kwargs):
    torch.cuda.set_device(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    f(*args, **kwargs)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


def cal_fun_t(n: int, device: str or torch.device or int, f: Callable, *args, **kwargs):
    if device == 'cpu':
        c_timer = cpu_timer
    else:
        c_timer = cuda_timer

    if n == 1:
        return c_timer(device, f, *args, **kwargs)

    # warm up
    c_timer(device, f, *args, **kwargs)

    t_list = []
    for _ in range(n * 2):
        t_list.append(c_timer(device, f, *args, **kwargs))
    t_list = np.asarray(t_list)
    return t_list[n:].mean()


def forward(x_seq, net: nn.Module):
    # assert not net.training
    with torch.no_grad():
        y_seq = net(x_seq)


def forward_backward(x_seq, net: nn.Module, optimizer: torch.optim.Optimizer):
    assert net.training
    optimizer.zero_grad()
    labels = torch.zeros(size=x_seq.shape[1:2]).long()
    y_seq = net(x_seq, labels=labels)
    err = losses.SpikeLoss().to(glv.rank)
    loss = get_loss(glv.network_config, err, y_seq, labels)
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    '''
    CUDA_VISIBLE_DEVICES=2 python test_speed.py --config networks/FMNIST.yaml
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str, help='run on which device, e.g., "cpu" or "cuda:0"')
    parser.add_argument("--repeats", default=32, type=int, help='the number of repeated experiments')
    parser.add_argument("--config", type=str, help="config file")
    args = parser.parse_args()

    device = args.device
    repeats = args.repeats
    glv.rank = int(args.device.split(':')[1])
    print(device, glv.rank)

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config

    params = parse(config_path)
    glv.init(params['Network'], params['Layers'] if 'Layers' in params.parameters else None)
    loss_type = params["Network"]["loss"]

    with open(f'./train_{params["Network"]["dataset"]}_{loss_type}.csv', 'w+') as csv_file:        
        T = params['Network']['t_train']
        print(T)
        N = params['Network']['batch_size']

        csv_file.write(f'repeats={args.repeats}, b={N}\n')
        csv_file.write(f'T, t_train, t_inference\n')

        x_seq = torch.rand([T, N, 1, 28, 28], device=device)

        net = cnns.Network(list(x_seq.shape[-3:])).to(device)
        
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-6)

        net.train()
        t_train = cal_fun_t(repeats, device, forward_backward, x_seq, net, optimizer)
        print(f't_train={t_train}')
        net.eval()
        t_inference = cal_fun_t(repeats, device, forward, x_seq, net)
        print(f't_inference={t_inference}')

        csv_file.write(f'{T}, {t_train}, {t_inference}\n')
