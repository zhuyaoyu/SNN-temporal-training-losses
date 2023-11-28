import torch
import torch.nn as nn
import torch.nn.functional as f
from layers.functions import neuron_forward, neuron_backward, bn_forward, readConfig, initialize
import global_v as glv
import torch.backends.cudnn as cudnn
from torch.utils.cpp_extension import load_inline, load
from torch.cuda.amp import custom_fwd, custom_bwd
from datetime import datetime

if torch.__version__ < "1.11.0":
    cpp_wrapper = load(name="cpp_wrapper", sources=["layers/cpp_wrapper.cpp"], verbose=True)
    conv_backward_input = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        cpp_wrapper.cudnn_convolution_backward_input(input.shape, grad_output, weight, padding, stride, dilation, groups,
                                                     cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)
    conv_backward_weight = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        cpp_wrapper.cudnn_convolution_backward_weight(weight.shape, grad_output, input, padding, stride, dilation, groups, 
                                                      cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)
else:
    bias_sizes, output_padding = [0, 0, 0, 0], [0, 0]
    transposed = False
    conv_backward_input = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        torch.ops.aten.convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, 
                                            transposed, output_padding, groups, [True, False, False])[0]
    conv_backward_weight = lambda grad_output, input, weight, padding, stride, dilation, groups: \
        torch.ops.aten.convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, 
                                            transposed, output_padding, groups, [False, True, False])[1]


class ConvLayer(nn.Conv2d):
    def __init__(self, config, name=None, groups=1):
        self.name = name
        threshold = config['threshold'] if 'threshold' in config else None
        self.type = 'conv'
        in_features = config['in_channels']
        out_features = config['out_channels']
        kernel_size = config['kernel_size']

        padding = config['padding'] if 'padding' in config else 0
        stride = config['stride'] if 'stride' in config else 1
        dilation = config['dilation'] if 'dilation' in config else 1

        self.kernel = readConfig(kernel_size, 'kernelSize')
        self.stride = readConfig(stride, 'stride')
        self.padding = readConfig(padding, 'stride')
        self.dilation = readConfig(dilation, 'stride')

        super(ConvLayer, self).__init__(in_features, out_features, self.kernel, self.stride, self.padding,
                                        self.dilation, groups, bias=False)
        self.weight = torch.nn.Parameter(self.weight.cuda(), requires_grad=True)
        self.norm_weight = torch.nn.Parameter(torch.ones(out_features, 1, 1, 1, device='cuda'))
        self.norm_bias = torch.nn.Parameter(torch.zeros(out_features, 1, 1, 1, device='cuda'))
        # self.threshold = torch.nn.Parameter(torch.ones(out_features, 1, 1, 1, device='cuda') * threshold)
        self.threshold = torch.ones(out_features, 1, 1, 1, device='cuda') * threshold
        
        glv.l_states.training.add_layer()
        glv.l_states.testing.add_layer()

        print('conv')
        print(f'Shape of weight is {list(self.weight.shape)}')  # Cout * Cin * Hk * Wk
        print(f'stride = {self.stride}, padding = {self.padding}, dilation = {self.dilation}, groups = {self.groups}')
        print("-----------------------------------------")

    def forward_pass(self, x):
        config_n = glv.network_config
        theta_m = 1 / config_n['tau_m']
        theta_s = 1 / config_n['tau_s']
        theta_grad = 1 / config_n['tau_grad'] if config_n['gradient_type'] == 'exponential' else -123456789  # instead of None
        
        normed_weight = bn_forward(self.weight, self.norm_weight, self.norm_bias)
        y = ConvFunc.apply(x, normed_weight, self.threshold, (self.bias, self.stride, self.padding, self.dilation, self.groups), (theta_m, theta_s, theta_grad))
        
        with torch.no_grad():
            spikeCnt = torch.sum(y.reshape(y.shape[0],-1),dim=1).detach().cpu().numpy()
            states = glv.l_states.training if self.training else glv.l_states.testing
            states.spikeTimeCnt[states.layerIdx] += spikeCnt
            states.layerIdx = (states.layerIdx + 1) % states.layerNum
        return y

    def forward(self, x):
        if glv.init_flag:
            x = initialize(self, x)
            self.th0 = torch.min(self.threshold)
            self.normw0 = torch.min(self.norm_weight)
            return x

        with torch.no_grad():
            if glv.network_config['norm_threshold']:
                assert(torch.min(self.norm_weight.data) > 0)
                # self.threshold.data /= self.norm_weight.data
                self.threshold = self.threshold / self.norm_weight.data
                self.norm_bias.data /= self.norm_weight.data
                self.norm_weight.data = torch.ones_like(self.norm_weight.data)
            self.weight_clipper()

        y = self.forward_pass(x)
        return y

    def weight_clipper(self):
        self.weight.data = self.weight.data.clamp(-4, 4)
        # self.threshold.data = self.threshold.data.clamp(self.th0 / 5, self.th0 * 5)
        # self.norm_weight.data = self.norm_weight.data.clamp(self.normw0 / 5, self.normw0 * 5)


class ConvFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, threshold, conv_config, neuron_config):
        # input.shape: T * n_batch * C_in * H_in * W_in
        bias, stride, padding, dilation, groups = conv_config
        T, n_batch, C, H, W = inputs.shape

        in_I = f.conv2d(inputs.reshape(T * n_batch, C, H, W), weight, bias, stride, padding, dilation, groups)
        _, C, H, W = in_I.shape
        in_I = in_I.reshape(T, n_batch, C, H, W)

        delta_u, delta_u_t, outputs = neuron_forward(in_I, threshold, neuron_config)

        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight, threshold)
        ctx.conv_config = conv_config

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        # shape of grad_delta: T * n_batch * C * H * W
        (delta_u, delta_u_t, inputs, outputs, weight, threshold) = ctx.saved_tensors
        bias, stride, padding, dilation, groups = ctx.conv_config
        grad_delta *= outputs
        # sum_next = grad_delta.sum().item()
        # print("Max of dLdt: ", abs(grad_delta).max().item())

        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)

        T, n_batch, C, H, W = grad_delta.shape
        inputs = inputs.reshape(T * n_batch, *inputs.shape[2:])
        grad_in_, grad_w_ = map(lambda x: x.reshape(T * n_batch, C, H, W), [grad_in_, grad_w_])
        
        grad_input = conv_backward_input(grad_in_, inputs, weight, padding, stride, dilation, groups) * inputs
        grad_weight = conv_backward_weight(grad_w_, inputs, weight, padding, stride, dilation, groups)

        # stats
        states = glv.l_states.training
        states.layerIdx = (states.layerIdx + states.layerNum - 1) % states.layerNum
        states.gradSumW[states.layerIdx] += torch.sum(grad_weight).item()

        # sum_last = grad_input.sum().item()
        # print(f'sum_next = {sum_next}, sum_last = {sum_last}')
        # assert(abs(sum_next - sum_last) < 1)
        
        return grad_input.reshape(T, n_batch, *inputs.shape[1:]), grad_weight, None, None, None
        '''
        lr = glv.network_config['lr_norm']
        norm_weight_1 = norm_weight #+ lr * grad_norm_w
        grad_th = -threshold / norm_weight_1 * grad_norm_w
        grad_norm_b = (grad_norm_b * norm_weight - norm_bias * grad_norm_w) / norm_weight_1
        return grad_input.reshape(T, n_batch, *inputs.shape[1:]), grad_weight, None, grad_norm_b, grad_th, None, None
        '''
