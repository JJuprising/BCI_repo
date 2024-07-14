# -*- coding: utf-8 -*-
# @Time: 2024/4/25 21:42
# @Author : Young
# @File : FBSSVEPformer.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np
import einops
from scipy import signal
import math
import argparse
import sys

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y
def complex_spectrum_features(segmented_data, FFT_PARAMS):
    sample_freq = FFT_PARAMS[0]
    time_len = FFT_PARAMS[1]
    resolution, start_freq, end_freq = 0.2, 8, 64
    NFFT = round(sample_freq / resolution)
    fft_index_start = int(round(start_freq / resolution))
    fft_index_end = int(round(end_freq / resolution)) + 1
    sample_point = int(sample_freq * time_len)
    fft_result = np.fft.fft(segmented_data, axis=-1, n=NFFT) / (sample_point / 2)
    real_part = np.real(fft_result[:, :, :, fft_index_start:fft_index_end - 1])
    imag_part = np.imag(fft_result[:, :, :, fft_index_start:fft_index_end - 1])
    features_data = np.concatenate([real_part, imag_part], axis=-1)
    return features_data


class PreNorm(nn.Module):
    def __init__(self, token_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, token_length, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_length, token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, token_num, token_length, kernal_length, dropout):
        super().__init__()

        self.att2conv = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2, groups=1),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.att2conv(x)
        return out


class Transformer(nn.Module):
    def __init__(self, depth, token_num, token_length, kernal_length, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(token_length, Attention(token_num, token_length, kernal_length, dropout=dropout)),
                PreNorm(token_length, FeedForward(token_length, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SSVEPformer(nn.Module):
    def __init__(self, depth, attention_kernal_length, chs_num, class_num, dropout):
        super().__init__()
        token_num = chs_num * 2
        token_dim = 560
        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(chs_num, token_num, 1, padding=1 // 2, groups=1),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.transformer = Transformer(depth, token_num, token_dim, attention_kernal_length, dropout)

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * token_num, class_num * 6),
            nn.LayerNorm(class_num * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(class_num * 6, class_num)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = self.transformer(x)
        return self.mlp_head(x)

class FB_SSVEPformer(nn.Module):
    # 需要修改build_model函数,需要分成3个子带
    def __init__(self, depth, attention_kernal_length, chs_num, class_num, dropout, num_subbands = 3):
        super().__init__()
        self.num_subbands = num_subbands
        # 创建多个 SSVEPformer 子网络
        self.subnetworks = nn.ModuleList([
            SSVEPformer(depth, attention_kernal_length, chs_num, class_num, dropout)
            for _ in range(num_subbands)
        ])
        # 结果融合层
        self.fusion_layer = nn.Conv1d(3, 1, kernel_size=1)

    def forward(self, x):
        # x:(30, 8, 3, 560)
        # 每个子网络处理对应的子带数据
        subband_outputs = [
            subnetwork(x[:, :, i, :]) # (30, 12)
            for i, subnetwork in enumerate(self.subnetworks)
        ]
        # 将子网络输出进行融合
        outputs = torch.stack(subband_outputs, dim=2) # (30, 12, 3)
        outputs = torch.transpose(outputs, 1, 2) # (30, 3, 12)
        fused_output = self.fusion_layer(outputs) # (30, 1, 12)
        return fused_output.squeeze(dim=1)
