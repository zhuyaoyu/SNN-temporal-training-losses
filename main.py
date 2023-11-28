import os
import sys
import shutil

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from network_parser import parse
from datasets import *
from datasets.utils import TTFS
import cnns
from stats import learningStats
import layers.losses as losses
import numpy as np
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
import global_v as glv

from sklearn.metrics import confusion_matrix
import argparse


log_interval = 100
multigpu = False


def get_loss(network_config, err, outputs, labels):
    T = network_config['n_steps']
    if network_config['loss'] in ['kernel', 'timing']:
        targets = torch.zeros_like(outputs)
        device = torch.device(glv.rank)
        # if T >= 8:
        if T >= 20:
            desired_spikes = torch.tensor([0, 1], device=device).repeat(T // 2)
            if T % 2 == 1:
                desired_spikes = torch.cat(
                    [torch.zeros(1, device=device), desired_spikes])
        else:
            desired_spikes = torch.ones(T, device=device)
            desired_spikes[0] = 0
        for i in range(len(labels)):
            targets[..., i, labels[i]] = desired_spikes

    if network_config['loss'] in ["count", "count+", "count0"]:
        # set target signal
        desired_count = network_config['desired_count']
        undesired_count = network_config['undesired_count']
        targets = torch.ones_like(outputs[0]) * undesired_count
        for i in range(len(labels)):
            targets[i, labels[i]] = desired_count
        if network_config['loss'] == "count":
            loss = err.spike_count(outputs, targets)
        elif network_config['loss'] == "count+":
            loss = err.spike_count_plus(outputs, targets)
        else:
            loss = err.spike_count_sum0(outputs, targets)
    elif network_config['loss'] == "kernel":
        loss = err.spike_kernel(outputs, targets)
    elif network_config['loss'] == "TET":
        # set target signal
        loss = err.spike_TET(outputs, labels)
    elif network_config['loss'] == "CE+":
        # set target signal
        loss = err.spike_CE_plus(outputs, labels)
    elif network_config['loss'] == "timing":
        loss = err.spike_timing(outputs, targets, labels)
    elif network_config['loss'] == "TTFS":
        loss = err.spike_TTFS(outputs, labels)
    elif network_config['loss'] == "TTFS0":
        loss = err.spike_TTFS0(outputs, labels)
    else:
        raise Exception('Unrecognized loss function.')

    return loss.to(glv.rank)


def readout(output, T):
    if glv.network_config['readout'] == 'TTFS':
        out = output * 2. ** (-torch.arange(T, device=torch.device(glv.rank)).reshape(T, 1, 1))
        # out = output * (1.1 - torch.arange(T, device=torch.device(glv.rank)).reshape(T, 1, 1) / T / 10)
    else:
        out = output * (1.1 - torch.arange(T, device=torch.device(glv.rank)).reshape(T, 1, 1) / T / 10)
    return torch.sum(out, dim=0).detach()


def preprocess(inputs, network_config):
    inputs = inputs.to(glv.rank)
    if network_config['encoding'] == 'TTFS':
        inputs = torch.stack([TTFS(data, T) for data in inputs], dim=0)
    if len(inputs.shape) < 5:
        inputs = inputs.unsqueeze_(0).repeat(T, 1, 1, 1, 1)
    else:
        inputs = inputs.permute(1, 0, 2, 3, 4)
    return inputs


def train(network, trainloader, opti, epoch, states, err):
    network.train()
    train_loss, correct, total = 0, 0, 0
    cnt_oneof, cnt_unique = 0, 0
    network_config = glv.network_config
    batch_size = network_config['batch_size']
    scaler = GradScaler()
    start_time = datetime.now()

    forward_time, backward_time, data_time, other_time, glv.time_use = 0, 0, 0, 0, 0
    t0 = datetime.now()
    num_batch = len(trainloader)
    batch_idx = 0
    for inputs, labels in trainloader:
        torch.cuda.synchronize()
        data_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()
        batch_idx += 1

        labels, inputs = (x.to(glv.rank) for x in (labels, inputs))
        inputs = preprocess(inputs, network_config)
        # forward pass
        if network_config['amp']:
            with autocast():
                outputs = network(inputs, labels, epoch, True)
                loss = get_loss(network_config, err, outputs, labels)
        else:
            outputs = network(inputs, labels, epoch, True)
            loss = get_loss(network_config, err, outputs, labels)
        assert (len(outputs.shape) == 3) # T * N * C

        torch.cuda.synchronize()
        forward_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()
        # backward pass
        opti.zero_grad()
        if network_config['amp']:
            scaler.scale(loss).backward()
            scaler.unscale_(opti)
            clip_grad_norm_(network.parameters(), 1000)
            scaler.step(opti)
            scaler.update()
        else:
            loss.backward()
            # print(sum([x.norm() for x in network.parameters()]))
            clip_grad_norm_(network.parameters(), 1000)
            opti.step()
        # (network.module if multigpu else network).weight_clipper()
        torch.cuda.synchronize()
        backward_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()

        spike_sum = readout(glv.outputs_raw, T)
        predicted = torch.argmax(spike_sum, axis=1)
        train_loss += torch.sum(loss).item()
        total += len(labels)
        correct += (predicted == labels).sum().item()

        states.training.correctSamples = correct
        states.training.numSamples = total
        states.training.lossSum += loss.to('cpu').data.item()

        labels = labels.reshape(-1)
        idx = torch.arange(labels.shape[0], device=torch.device(glv.rank))
        nspike_label = spike_sum[idx, labels]
        cnt_oneof += torch.sum(nspike_label ==
                               torch.max(spike_sum, axis=1).values).item()
        spike_sum[idx, labels] -= 1e-3
        cnt_unique += torch.sum(nspike_label >
                                torch.max(spike_sum, axis=1).values).item()
        spike_sum[idx, labels] += 1e-3

        if (not multigpu or dist.get_rank() == 0) and (batch_idx % log_interval == 0 or batch_idx == num_batch):
            # if batch_idx % log_interval == 0:
            states.print(epoch, batch_idx,
                         (datetime.now() - start_time).total_seconds())
            print('Time consumed on loading data = %.2f, forward = %.2f, backward = %.2f, other = %.2f'
                  % (data_time, forward_time, backward_time, other_time))
            # print('Time consumed on recorded interval = %.2f' % (glv.time_use))
            data_time, forward_time, backward_time, other_time, glv.time_use = 0, 0, 0, 0, 0

            spike_cnt = dict()
            spike_count = torch.sum(glv.outputs_raw, dim=0).detach()
            for i in range(6):
                spike_cnt['nspike=' + str(i)] = (torch.sum(spike_count > i - 1+1e-5) - torch.sum(
                    spike_count > i+1e-5) if i < 5 else torch.sum(spike_count > 4+1e-5)).item()
            # print('Spike_cnt is', spike_cnt)

            avg_oneof, avg_unique = cnt_oneof / \
                (batch_size * batch_idx), cnt_unique / (batch_size * batch_idx)
            print('Percentage of partially right = %.2f%%, entirely right = %.2f%%' % (
                avg_oneof * 100, avg_unique * 100))
            print()
        torch.cuda.synchronize()
        other_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()

    acc = correct / total
    train_loss = train_loss / total

    return acc, train_loss


def test(network, testloader, epoch, states, log_dir):
    network.eval()
    global best_acc
    correct = 0
    total = 0
    tot_inference_time = 0
    network_config = glv.network_config
    T = network_config['n_steps']
    n_class = network_config['n_class']
    time = datetime.now()
    y_pred = []
    y_true = []
    num_batch = len(testloader)
    batch_idx = 0
    for inputs, labels in testloader:
        batch_idx += 1
        inputs = preprocess(inputs, network_config)
        # forward pass
        labels = labels.to(glv.rank)
        inputs = inputs.to(glv.rank)
        with torch.no_grad():
            outputs = network(inputs, None, epoch, False)

        spike_sum = readout(outputs, T).cpu().numpy()
        predicted = np.argmax(spike_sum, axis=1)
        labels = labels.cpu().numpy()
        y_pred.append(predicted)
        y_true.append(labels)
        total += len(labels)
        correct += (predicted == labels).sum().item()

        # get early exit time
        # outputs: T * N * C
        T, N, C = outputs.shape
        sorted_outputs, _ = torch.sort(outputs, dim=2, descending=True)
        sorted_outputs = sorted_outputs.cpu().numpy()
        times = np.argmax(sorted_outputs[:,:,0] != sorted_outputs[:,:,1], axis=0)
        outputs = outputs.cpu().numpy()
        times[outputs[times, np.arange(N), predicted] == False] = T
        times[predicted != labels] = T
        tot_inference_time += sum(times)

        states.testing.correctSamples += (predicted == labels).sum().item()
        states.testing.numSamples = total
        if batch_idx % log_interval == 0 or batch_idx == num_batch:
            states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())
    print()

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    nums = np.bincount(y_true)
    confusion = confusion_matrix(
        y_true, y_pred, labels=np.arange(n_class)) / nums.reshape(-1, 1)

    test_acc = correct / total
    avg_inference_time = tot_inference_time / total

    state = {
        'net': (network.module if multigpu else network).state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(log_dir, 'last.pth'))

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(state, os.path.join(log_dir, 'best.pth'))
    return test_acc, confusion, avg_inference_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store',
                        dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint',
                        help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-seed', type=int, default=3,
                        help='random seed (default: 3)')
    parser.add_argument('-dist', type=str, default="nccl",
                        help='distributed data parallel backend')
    parser.add_argument('--local_rank', type=int, default=-1)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config

    params = parse(config_path)

    # check GPU
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)
    # set GPU
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.dist)
        glv.rank = args.local_rank
        multigpu = True
    else:
        glv.rank = 0
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    glv.init(params['Network'], params['Layers']
             if 'Layers' in params.parameters else None)

    data_path = os.path.expanduser(params['Network']['data_path'])
    dataset_func = {"MNIST": loadMNIST.get_mnist,
                    "NMNIST": loadSpiking.get_nmnist,
                    "FashionMNIST": loadFashionMNIST.get_fashionmnist,
                    "CIFAR10": loadCIFAR10.get_cifar10,
                    "CIFAR100": loadCIFAR100.get_cifar100,
                    "DVS128Gesture": loadSpiking.get_dvs128_gesture,
                    "CIFAR10DVS": loadSpiking.get_cifar10_dvs,
                    "SMNIST": loadSMNIST.get_smnist}
    try:
        datasets = dataset_func[params['Network']['dataset']](data_path, params['Network'])
        if len(datasets) == 2:
            trainset, testset = datasets
            train_sampler, collate_fn = None, None
        elif len(datasets) == 4:
            trainset, testset, train_sampler, collate_fn = datasets
        else:
            raise Exception
    except:
        raise Exception('Unrecognized dataset name.')
    batch_size = params['Network']['batch_size']
    if multigpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset)
    if train_sampler is None:
        train_loader = torch.utils.data.DataLoader(
            trainset, shuffle=True, batch_size=batch_size, num_workers=8, collate_fn=collate_fn, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset, sampler=train_sampler, batch_size=batch_size, num_workers=8, collate_fn=collate_fn, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, num_workers=8, collate_fn=collate_fn, pin_memory=True)

    if 'model_import' not in glv.network_config:
        net = cnns.Network(
            list(train_loader.dataset[0][0].shape[-3:])).to(glv.rank)
    else:
        exec(f"from {glv.network_config['model_import']} import Network")
        net = Network().to(glv.rank)
        print(net)

    T = params['Network']['t_train']
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
        net.load_state_dict(checkpoint['net'])
        epoch_start = checkpoint['epoch'] + 1
        print('Network loaded.')
        print(f'Start training from epoch {epoch_start}.')
    else:
        inputs = torch.stack([train_loader.dataset[i][0]
                             for i in range(batch_size)], dim=0).to(glv.rank)
        inputs = preprocess(inputs, glv.network_config)
        # print("Start to initialize.")
        # initialize weights
        with torch.no_grad():
            glv.init_flag = True
            net(inputs, None, None, False)
            glv.init_flag = False
        epoch_start = 1

    error = losses.SpikeLoss().to(glv.rank)  # the loss is not defined here
    if multigpu:
        net = DDP(net, device_ids=[glv.rank], output_device=glv.rank)
    optim_type, weight_decay, weight_decay_norm, lr, lr_norm = (
        glv.network_config[x] for x in ('optimizer', 'weight_decay', 'weight_decay_norm', 'lr', 'lr_norm'))
    assert(optim_type in ['SGD', 'Adam', 'AdamW'])

    param, norm_param, th_param = [], [], []
    for layer in net.modules():
        if layer.type in ['conv', 'linear']:
            param.append(layer.weight)
            norm_param.extend([layer.norm_weight, layer.norm_bias])
            # norm_param.extend([layer.norm_bias])
            # th_param.extend([layer.threshold])

    optim_dict = {'SGD': torch.optim.SGD,
                  'Adam': torch.optim.Adam,
                  'AdamW': torch.optim.AdamW}
    optimizer = optim_dict[optim_type]([
        {'params': param},
        {'params': norm_param, 'lr': lr_norm, 'weight_decay': weight_decay_norm},
        # {'params': th_param, 'lr': lr_norm, 'weight_decay': -weight_decay_norm},
    ], lr=lr, weight_decay=weight_decay)
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=glv.network_config['epochs'])

    best_acc = 0
    l_states = glv.l_states

    log_dir = f"{params['Network']['log_path']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
    writer = SummaryWriter(log_dir)
    confu_mats = []
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.split(config_path)[-1]))
    def ignore_func(dirname, filenames):
        suffix_list = ['py', 'cpp', 'cu', 'm']
        ignore_file = [name for name in filenames if name.find('.') != -1
                    and not (name.split('.')[-1] in suffix_list)]
        return ignore_file + ['__pycache__', 'logs']
    # copy code into the log dir, no longer worry about forgetting code change!
    shutil.copytree('.', os.path.join(log_dir, '_code'), ignore=ignore_func, dirs_exist_ok=True)

    (net.module if multigpu else net).train()
    for epoch in range(epoch_start, params['Network']['epochs'] + epoch_start):
        if multigpu:
            train_loader.sampler.set_epoch(epoch)
        l_states.training.reset()
        train_acc, loss = train(
            net, train_loader, optimizer, epoch, l_states, error)
        l_states.training.update()
        l_states.testing.reset()
        test_acc, confu_mat, infer_time = test(net, test_loader, epoch, l_states, log_dir)
        l_states.testing.update()
        lr_scheduler.step()

        confu_mats.append(confu_mat)
        if glv.rank == 0:
            writer.add_scalars('Accuracy', {'train': train_acc,
                                            'test': test_acc}, epoch)
            writer.add_scalars('Loss', {'loss': loss}, epoch)
            writer.add_scalars('Sum_grad_t', {'sum_grad_t': l_states.training.gradSumT,
                                              'sum_abs_grad_t': l_states.training.gradAbsSumT}, epoch)
            writer.add_scalars('Grad_positive_ratio', {
                               'Grad_positive_ratio': l_states.training.gradSumT / l_states.training.gradAbsSumT}, epoch)
            writer.add_scalars('Inference_time', {'Inference_time': infer_time}, epoch)

            idx, convIdx, linearIdx = 0, 0, 0
            for layer in net.modules():
                if layer.type in ['conv', 'linear']:
                    if layer.type=='conv':
                        name = f'conv{convIdx}'
                        convIdx += 1
                    else:
                        name = f'linear{linearIdx}'
                        linearIdx += 1
                    writer.add_histogram(tag=name+'_weight', values=layer.weight.data, global_step=epoch)
                    writer.add_histogram(tag=name+'_norm_weight', values=layer.norm_weight.data, global_step=epoch)
                    writer.add_histogram(tag=name+'_norm_bias', values=layer.norm_bias.data, global_step=epoch)
                    writer.add_histogram(tag=name+'_threshold', values=layer.threshold, global_step=epoch)
                    writer.add_scalars('Sum_grad_w', {name+'_sum_grad_w': l_states.training.gradSumW[idx]}, epoch)
                    idx += 1
            
            np.save(log_dir + 'confusion_matrix.npy', np.stack(confu_mats))
            np.save(log_dir + 'train_spike_timing.npy', np.stack(l_states.training.spikeTimeLog))
            np.save(log_dir + 'test_spike_timing.npy', np.stack(l_states.testing.spikeTimeLog))
