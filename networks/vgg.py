import torch
import torch.nn as nn
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.linear as linear
from torch.cuda.amp import custom_fwd, custom_bwd
import global_v as glv


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    config = {'in_channels': in_planes, 'out_channels': out_planes,
              'kernel_size': 3, 'padding': 1, 'stride': stride, 'dilation': dilation, 'threshold': 1}
    return conv.ConvLayer(config=config, name=None)


class SpikingVGG(nn.Module):
    def __init__(self, layers, num_classes=10, groups=1, input_shape=[32,32], norm_layer=None, **kwargs):
        super(SpikingVGG, self).__init__()
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1

        self.groups = groups
        self.H, self.W = input_shape

        config = {'in_channels': 3, 'out_channels': self.inplanes,
                  'kernel_size': 3, 'padding': 1, 'threshold': 1}
        self.conv1 = conv.ConvLayer(config=config, name=None)

        self.layer1 = self._make_layer(128, layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(256, layers[1], stride=1, **kwargs)
        self.layer3 = self._make_layer(512, layers[2], stride=1, **kwargs)
        
        config = {'n_inputs': 512 * self.H * self.W, 'n_outputs': 2048, 'threshold': 1}
        self.fc1 = nn.Sequential(linear.LinearLayer(config=config, name=None),
                                 dropout.DropoutLayer(config={'p': 0.1}, name=None))
        config = {'n_inputs': 2048, 'n_outputs': 2048, 'threshold': 1}
        self.fc2 = nn.Sequential(linear.LinearLayer(config=config, name=None),
                                 dropout.DropoutLayer(config={'p': 0.1}, name=None))
        config = {'n_inputs': 2048, 'n_outputs': num_classes, 'threshold': 1}
        self.output = linear.LinearLayer(config=config, name=None)

    def _make_layer(self, planes, blocks, stride=1, **kwargs):
        layers = []
        layers.append(conv3x3(self.inplanes, planes, stride, self.groups, self.dilation))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(conv3x3(self.inplanes, planes, stride, self.groups, self.dilation))
        layers.append(pooling.PoolLayer(
            config={'kernel_size': 2}, name=None))
        layers.append(dropout.DropoutLayer(config={'p': 0.1}, name=None))
        self.H, self.W = self.H // 2, self.W // 2

        return nn.Sequential(*layers)

    def forward(self, x, labels, epoch, is_train):
        assert (is_train or labels == None)
        # See note [TorchScript super()]
        x = self.conv1(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.output(x, labels)

        return x


class Network(SpikingVGG):
    def __init__(self, input_shape=None):
        super(Network, self).__init__([1, 3, 3], glv.network_config['n_class'])
        print("-----------------------------------------")

