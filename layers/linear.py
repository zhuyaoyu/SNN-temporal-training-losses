import torch
import torch.nn as nn
import global_v as glv
from layers.functions import neuron_forward, neuron_backward, bn_forward, initialize
from torch.cuda.amp import custom_fwd, custom_bwd


class LinearLayer(nn.Linear):
    def __init__(self, config, name=None):
        # extract information for kernel and inChannels
        in_features = config['n_inputs']
        out_features = config['n_outputs']
        threshold = config['threshold'] if 'threshold' in config else None
        self.name = name
        self.type = 'linear'
        # self.in_shape = in_shape
        # self.out_shape = [out_features, 1, 1]

        if type(in_features) == int:
            n_inputs = in_features
        else:
            raise Exception('inFeatures should not be more than 1 dimension. It was: {}'.format(in_features.shape))
        if type(out_features) == int:
            n_outputs = out_features
        else:
            raise Exception('outFeatures should not be more than 1 dimension. It was: {}'.format(out_features.shape))

        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=False)
        self.weight = torch.nn.Parameter(self.weight.cuda(), requires_grad=True)
        self.norm_weight = torch.nn.Parameter(torch.ones(out_features,1, device='cuda'))
        self.norm_bias = torch.nn.Parameter(torch.zeros(out_features,1, device='cuda'))
        # self.threshold = torch.nn.Parameter(torch.ones(out_features, 1, device='cuda') * threshold)
        self.threshold = torch.ones(out_features, 1, device='cuda') * threshold

        glv.l_states.training.add_layer()
        glv.l_states.testing.add_layer()

        print("linear")
        print(self.name)
        # print(self.in_shape)
        # print(self.out_shape)
        print(f'Shape of weight is {list(self.weight.shape)}')
        print("-----------------------------------------")

    def forward_pass(self, x, labels=None):
        ndim = len(x.shape)
        assert (ndim == 3 or ndim == 5)
        if ndim == 5:
            T, n_batch, C, H, W = x.shape
            x = x.view(T, n_batch, C * H * W)
        config_n = glv.network_config
        theta_m = 1 / config_n['tau_m']
        theta_s = 1 / config_n['tau_s']
        theta_grad = 1 / config_n['tau_grad'] if config_n['gradient_type'] == 'exponential' else -123456789  # instead of None
        
        normed_weight = bn_forward(self.weight, self.norm_weight, self.norm_bias)
        y = LinearFunc.apply(x, normed_weight, self.threshold, (theta_m, theta_s, theta_grad), labels)
        
        with torch.no_grad():
            spikeCnt = torch.sum(y.reshape(y.shape[0], -1), dim=1).detach().cpu().numpy()
            states = glv.l_states.training if self.training else glv.l_states.testing
            states.spikeTimeCnt[states.layerIdx] += spikeCnt
            states.layerIdx = (states.layerIdx + 1) % states.layerNum
        return y

    def forward(self, x, labels=None):
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

        y = self.forward_pass(x, labels)
        return y

    def weight_clipper(self):
        self.weight.data = self.weight.data.clamp(-4, 4)
        # self.threshold.data = self.threshold.data.clamp(self.th0 / 5, self.th0 * 5)
        # self.norm_weight.data = self.norm_weight.data.clamp(self.normw0 / 5, self.normw0 * 5)


class LinearFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, threshold, config, labels):
        #input.shape: T * n_batch * N_in
        in_I = torch.matmul(inputs, weight.t())

        T, n_batch, N = in_I.shape
        theta_m, theta_s, theta_grad = config
        assert (theta_m != theta_s)
        delta_u, delta_u_t, outputs = neuron_forward(in_I, threshold, config)

        if labels is not None:
            glv.outputs_raw = outputs.clone()
            i2 = torch.arange(n_batch)
            # Add supervisory signal when synaptic potential is increasing:
            is_inc = (delta_u[:, i2, labels] > 0.05).float()
            _, i1 = torch.max(is_inc * torch.arange(1, T+1, device=is_inc.device).unsqueeze(-1), dim=0)
            outputs[i1, i2, labels] = (delta_u[i1, i2, labels] != 0).to(outputs)

        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight, threshold)
        ctx.is_out_layer = labels != None

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        # shape of grad_delta: T * n_batch * N_out

        (delta_u, delta_u_t, inputs, outputs, weight, threshold) = ctx.saved_tensors
        grad_delta *= outputs
        # sum_next = grad_delta.sum().item()
        # print("Max of dLdt: ", abs(grad_delta).max().item())

        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)

        grad_input = torch.matmul(grad_in_, weight) * inputs
        grad_weight = torch.sum(torch.matmul(grad_w_.transpose(1, 2), inputs), dim=0)
        
        # stats
        states = glv.l_states.training
        states.layerIdx = (states.layerIdx + states.layerNum - 1) % states.layerNum
        states.gradSumW[states.layerIdx] += torch.sum(grad_weight).item()

        # sum_last = grad_input.sum().item()
        # print(f'sum_next = {sum_next}, sum_last = {sum_last}')
        # assert(abs(sum_next - sum_last) < 1)

        if ctx.is_out_layer:
            glv.l_states.training.gradSumT += torch.sum(grad_delta)
            glv.l_states.training.gradAbsSumT += torch.sum(torch.abs(grad_delta))
        
        return grad_input, grad_weight, None, None, None
        '''
        lr = glv.network_config['lr_norm']
        norm_weight_1 = norm_weight #+ lr * grad_norm_w
        grad_th = -threshold / norm_weight_1 * grad_norm_w
        grad_norm_b = (grad_norm_b * norm_weight - norm_bias * grad_norm_w) / norm_weight_1
        return grad_input, grad_weight, None, grad_norm_b, grad_th, None, None
        '''
