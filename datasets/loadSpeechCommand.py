﻿import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms
from torchaudio.transforms import Spectrogram

from spikingjelly.datasets.speechcommands import SPEECHCOMMANDS
from scipy.signal import savgol_filter

from sklearn.metrics import confusion_matrix

import numpy as np

import math
import time
import argparse
from typing import Optional
from tqdm import tqdm

label_dict = {'yes': 0, 'stop': 1, 'no': 2, 'right': 3, 'up': 4, 'left': 5, 'on': 6, 'down': 7, 'off': 8, 'go': 9, 'bed': 10, 'three': 10, 'one': 10, 'four': 10, 'two': 10, 'five': 10, 'cat': 10, 'dog': 10, 'eight': 10, 'bird': 10, 'happy': 10, 'sheila': 10, 'zero': 10, 'wow': 10, 'marvin': 10, 'house': 10, 'six': 10, 'seven': 10, 'tree': 10, 'nine': 10, '_silence_': 11}
label_cnt = len(set(label_dict.values()))
n_mels = 40
f_max = 4000
f_min = 20
delta_order = 0
size = 16000
try:
    import cupy
    backend = 'cupy'
except ModuleNotFoundError:
    backend = 'torch'
    print('Cupy is not intalled. Using torch backend for neurons.')

def mel_to_hz(mels, dct_type):
    if dct_type == 'htk':
        return 700.0 * (10 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if torch.is_tensor(mels) and mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * \
            torch.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * math.exp(logstep * (mels - min_log_mel))

    return freqs


def hz_to_mel(frequencies, dct_type):
    if dct_type == 'htk':
        if torch.is_tensor(frequencies) and frequencies.ndim:
            return 2595.0 * torch.log10(1.0 + frequencies / 700.0)
        return 2595.0 * math.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if torch.is_tensor(frequencies) and frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + \
            torch.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + math.log(frequencies / min_log_hz) / logstep

    return mels


def create_fb_matrix(
        n_freqs: int,
        f_min: float,
        f_max: float,
        n_mels: int,
        sample_rate: int,
        dct_type: Optional[str] = 'slaney') -> Tensor:

    if dct_type != "htk" and dct_type != "slaney":
        raise ValueError("DCT type must be either 'htk' or 'slaney'")

    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f)
    m_min = hz_to_mel(f_min, dct_type)
    m_max = hz_to_mel(f_max, dct_type)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel)
    f_pts = mel_to_hz(m_pts, dct_type)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    # (n_freqs, n_mels + 2)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    if dct_type == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    return fb


class MelScaleDelta(nn.Module):
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self,
                 order,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 n_stft: Optional[int] = None,
                 dct_type: Optional[str] = 'slaney') -> None:
        super(MelScaleDelta, self).__init__()
        self.order = order
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.dct_type = dct_type

        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(
            f_min, self.f_max)

        fb = torch.empty(0) if n_stft is None else create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.dct_type)
        self.register_buffer('fb', fb)

    def forward(self, specgram: Tensor) -> Tensor:
        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        if self.fb.numel() == 0:
            tmp_fb = create_fb_matrix(specgram.size(
                1), self.f_min, self.f_max, self.n_mels, self.sample_rate, self.dct_type)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(
            specgram.transpose(1, 2), self.fb).transpose(1, 2)

        # unpack batch
        mel_specgram = mel_specgram.reshape(
            shape[:-2] + mel_specgram.shape[-2:]).squeeze()

        M = torch.max(torch.abs(mel_specgram))
        if M > 0:
            feat = torch.log1p(mel_specgram/M)
        else:
            feat = mel_specgram

        feat_list = [feat.numpy().T]
        for k in range(1, self.order + 1):
            feat_list.append(savgol_filter(
                feat.numpy(), 9, deriv=k, axis=-1, mode='interp', polyorder=k).T)

        return torch.as_tensor(np.expand_dims(np.stack(feat_list), axis=0))


class Pad(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, wav):
        wav_size = wav.shape[-1]
        pad_size = (self.size - wav_size) // 2
        padded_wav = torch.nn.functional.pad(
            wav, (pad_size, self.size-wav_size-pad_size), mode='constant', value=0)
        return padded_wav


class Rescale(object):

    def __call__(self, input):
        std = torch.std(input, axis=2, keepdims=True, unbiased=False) # Numpy std is calculated via the Numpy's biased estimator. https://github.com/romainzimmer/s2net/blob/82c38bf80b55d16d12d0243440e34e52d237a2df/data.py#L201 
        std.masked_fill_(std == 0, 1)

        return input / std

def collate_fn(data):

    X_batch = torch.cat([d[0] for d in data])
    std = X_batch.std(axis=(0, 2), keepdim=True, unbiased=False)
    X_batch.div_(std)

    y_batch = torch.tensor([d[1] for d in data])

    return X_batch, y_batch

def get_speech_command(data_path, network_config):
    sr = network_config['sample_rate']
    n_fft = int(30e-3*sr) # 48
    hop_length = int(10e-3*sr) # 16
    dataset_dir = data_path
    batch_size = network_config['batch_size']

    pad = Pad(size)
    spec = Spectrogram(n_fft=n_fft, hop_length=hop_length)
    melscale = MelScaleDelta(order=delta_order, n_mels=n_mels,
                             sample_rate=sr, f_min=f_min, f_max=f_max, dct_type='slaney')
    rescale = Rescale()

    transform = torchvision.transforms.Compose([pad,
                                                spec,
                                                melscale,
                                                rescale])

    print(label_cnt)
    print(dataset_dir)

    train_dataset = SPEECHCOMMANDS(
        label_dict, dataset_dir, silence_cnt=2300, url="speech_commands_v0.01", split="train", transform=transform, download=False)
    train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights))
    test_dataset = SPEECHCOMMANDS(
        label_dict, dataset_dir, silence_cnt=260, url="speech_commands_v0.01", split="test", transform=transform, download=False)
    
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16,
    #                               sampler=train_sampler, collate_fn=collate_fn)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=16, collate_fn=collate_fn, shuffle=False,
    #                              drop_last=False)
    
    return train_dataset, test_dataset, train_sampler, collate_fn