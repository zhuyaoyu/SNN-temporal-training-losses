import torch
import torch.nn.functional as f
import global_v as glv
from torch.cuda.amp import custom_fwd, custom_bwd
from math import sqrt


def psp(inputs):
    n_steps = glv.network_config['n_steps']
    tau_s = glv.network_config['tau_s']
    syns = torch.zeros_like(inputs).to(glv.rank)
    syn = torch.zeros(syns.shape[1:]).to(glv.rank)

    for t in range(n_steps):
        syn = syn * (1 - 1 / tau_s) + inputs[t, ...]
        syns[t, ...] = syn / tau_s
    return syns


class SpikeLoss(torch.nn.Module):
    """
    This class defines different spike based loss modules that can be used to optimize the SNN.
    """

    def __init__(self):
        super(SpikeLoss, self).__init__()

    def spike_count(self, output, target):
        # shape of output: T * N * C
        delta = loss_count.apply(output, target)
        return 1 / 2 * torch.sum(delta ** 2)

    def spike_count_plus(self, output, target):
        return loss_count_plus.apply(output, target)
    
    def spike_count_sum0(self, output, target):
        return loss_count_sum0.apply(output, target)

    def spike_kernel(self, output, target):
        # shape of output: T * N * C
        out = grad_sign.apply(output)
        delta = psp(out - target)
        return 1 / 2 * torch.sum(delta ** 2)

    def spike_TET(self, output, label):
        output = output.permute(1, 2, 0)
        out = grad_sign.apply(output)
        return f.cross_entropy(out, label.unsqueeze(-1).repeat(1, out.shape[-1]))

    def spike_CE_plus(self, output, label):
        # shape of output: T * N * C
        T, N, C = output.shape
        out_count = loss_CE_plus.apply(output, label)
        return T / 2 * f.cross_entropy(out_count, label)

    def spike_timing(self, output, target, label):
        output, target = output.permute(1, 2, 0), target.permute(1, 2, 0)
        delta = loss_spike_timing.apply(output, target, label)
        return 1 / 2 * torch.sum(delta ** 2)
    
    def spike_TTFS(self, output, label):
        T, N, C = output.shape
        timing = loss_TTFS.apply(output)
        return T * f.cross_entropy(timing, label)
    
    def spike_TTFS0(self, output, label):
        T, N, C = output.shape
        timing = loss_TTFS0.apply(output)
        return T * f.cross_entropy(timing, label)


class loss_count(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, output, target):
        desired_count = glv.network_config['desired_count']
        undesired_count = glv.network_config['undesired_count']
        T = output.shape[0]
        out_count = torch.sum(output, dim=0)
        out_count[(target == desired_count) & (
            out_count > desired_count)] = desired_count
        out_count[(target == undesired_count) & (
            out_count < undesired_count)] = undesired_count

        delta = (out_count - target) / T
        delta = delta.unsqueeze_(0).repeat(T, 1, 1)
        ctx.save_for_backward(output, out_count, target)
        return delta

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        desired_count = glv.network_config['desired_count']
        output, out_count, target = ctx.saved_tensors
        T, N, C = output.shape
        desired_count = glv.network_config['desired_count']
        ratio = (torch.sum(out_count[target != desired_count]) + 1e-5) / \
            (torch.sum(out_count[target == desired_count]) + 1e-5)
        mask = (target == desired_count).unsqueeze(0).repeat(T,1,1)
        grad = grad * output
        grad[mask] = grad[mask] * max(ratio / 10, 1)
        return -grad, None


class loss_count_plus(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, output, target):
        desired_count = glv.network_config['desired_count']
        undesired_count = glv.network_config['undesired_count']
        T = output.shape[0]
        out_count = torch.sum(output, dim=0)
        out_count[(target == desired_count) & (
            out_count > desired_count)] = desired_count
        out_count[(target == undesired_count) & (
            out_count < undesired_count)] = undesired_count

        delta = (out_count - target) / T
        ctx.save_for_backward(output, out_count, target, delta)
        delta = delta.unsqueeze_(0).repeat(T, 1, 1)
        return 1 / 2 * torch.sum(delta ** 2)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        (output, out_count, target, delta) = ctx.saved_tensors
        T, N, C = output.shape
        delta = delta.reshape(target.shape)
        desired_count = glv.network_config['desired_count']
        ratio = (torch.sum(out_count[target != desired_count]) + 1e-5) / \
            (torch.sum(out_count[target == desired_count]) + 1e-5)

        mask = target == desired_count
        delta[mask] = delta[mask] * max(ratio / 10, 1)
        out_count_inv = 1 / out_count
        out_count_inv[out_count == 0] = 0
        delta = delta * out_count_inv
        delta = delta.unsqueeze_(0).repeat(T, 1, 1) * output
        
        sign = -1
        return sign * delta, None


class loss_CE_plus(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, output, label):
        # shape of output: T * N * C
        T = output.shape[0]
        out_count = torch.sum(output, dim=0)
        out_count = out_count / T * 2
        out_count[out_count == 0] = -20

        ctx.save_for_backward(output, label)
        return out_count

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        output, label = ctx.saved_tensors

        T, N, C = output.shape
        
        out_count = torch.sum(output, dim=0)
        # if the correct neuron does not emit any spikes, eliminate all gradients on that sample
        grad[out_count[torch.arange(N), label] == 0, :] = 0
        out_count[out_count == 0] = -20
        grad = grad / out_count
        grad = grad.unsqueeze_(0).repeat(T, 1, 1)
        grad = grad * output

        sign = -1
        return sign * grad, None


class grad_sign(torch.autograd.Function):  # a and u is the increment of each time steps
    @staticmethod
    @custom_fwd
    def forward(ctx, outputs):
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        sign = -1
        return sign * grad


def spike_to_timing(x):
    N, C, T = x.shape
    nspike = torch.sum(x, dim=-1).long()
    idx = torch.arange(T, device=x.device).reshape(
        1, 1, T).repeat(N, C, 1)
    idx = idx < nspike.unsqueeze_(-1)

    times = x * (torch.arange(1, T + 1, device=x.device).reshape(1, T))
    times[idx] = times[x.bool()]
    times[~idx] = T + 1
    return times


class loss_spike_timing(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, output, target, label):
        # shape of output: N * C * T
        N, C, T = output.shape
        out_timing, tar_timing = (spike_to_timing(x) for x in (output, target))

        delta = (out_timing - tar_timing + 0.5) / T
        correct = torch.zeros_like(delta).bool()
        correct[torch.arange(N, device=label.device), label, :] = 1
        delta[correct & (delta < 0) | ~correct & (delta > 0)] = 0
        delta[~correct & (delta < 0)] = -1 / sqrt(C)

        ctx.save_for_backward(out_timing, tar_timing, output)
        return delta

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # grad == delta
        out_timing, tar_timing, output = ctx.saved_tensors
        N, C, T = output.shape
        grad[out_timing > T] = 0

        # grad in out_timing -> grad in output
        nspike = torch.sum(output, dim=-1).long()
        idx = torch.arange(T, device=output.device).reshape(
            1, 1, T).repeat(N, C, 1)
        idx = idx < nspike.unsqueeze_(-1)

        grad_out = torch.zeros_like(output)
        grad_out[output.bool()] = grad[idx]

        return grad_out, None, None


class loss_count_sum0(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, output, target):
        desired_count = glv.network_config['desired_count']
        undesired_count = glv.network_config['undesired_count']
        T = output.shape[0]
        out_count = torch.sum(output, dim=0)
        out_count[(target == desired_count) & (
            out_count > desired_count)] = desired_count
        out_count[(target == undesired_count) & (
            out_count < undesired_count)] = undesired_count

        delta = (out_count - target) / T
        ctx.save_for_backward(output, out_count, target, delta)
        delta = delta.unsqueeze_(0).repeat(T, 1, 1)
        return 1 / 2 * torch.sum(delta ** 2)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        (output, out_count, target, delta) = ctx.saved_tensors
        T, N, C = output.shape
        delta = delta.reshape(target.shape)
        desired_count = glv.network_config['desired_count']
        ratio = (torch.sum(out_count[target != desired_count]) + 1e-5) / \
            (torch.sum(out_count[target == desired_count]) + 1e-5)

        mask = target == desired_count
        delta[mask] = delta[mask] * max(ratio / 10, 1)
        out_count_inv = 1 / out_count
        out_count_inv[out_count == 0] = 0
        delta = delta * out_count_inv
        delta = delta.unsqueeze_(0).repeat(T, 1, 1) * output
        
        # Weight balancing
        delta[output.bool()] -= torch.sum(delta) / torch.sum(output)
        
        sign = -1
        return sign * delta, None


class loss_TTFS(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, output):
        # shape of output: T * N * C
        T, N, C = output.shape
        value, timing = torch.max(output, dim=0)
        timing = 1 - timing / T
        timing[value == 0] = -3
        # print(value, timing)

        ctx.save_for_backward(output)
        return timing

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # grad == delta
        output = ctx.saved_tensors[0]
        T, N, C = output.shape
        timing = torch.max(output, dim=0).indices
        i1, i2, i3 = timing.reshape(-1).long(), torch.arange(N*C) // C, torch.arange(N*C) % C

        grad_out = torch.zeros_like(output)
        grad_out[i1, i2, i3] = grad.reshape(-1)
        grad_out[output==0] = 0

        return -grad_out


class loss_TTFS0(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, output):
        # shape of output: T * N * C
        T, N, C = output.shape
        value, timing = torch.max(output, dim=0)
        timing = 1 - timing / T
        timing[value == 0] = -20
        # print(value, timing)

        ctx.save_for_backward(output)
        return timing

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # grad == delta
        output = ctx.saved_tensors[0]
        T, N, C = output.shape
        timing = torch.max(output, dim=0).indices
        i1, i2, i3 = timing.reshape(-1).long(), torch.arange(N*C) // C, torch.arange(N*C) % C

        grad_out = torch.zeros_like(output)
        grad_out[i1, i2, i3] = grad.reshape(-1)
        grad_out[output==0] = 0

        return -grad_out