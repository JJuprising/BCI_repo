# Designer:Yudong Pan
# Coder:God's hand
# Time:2024/2/3 23:45
import torch
import torch.nn as nn
import numpy as np
import einops
from scipy import signal
import math
import argparse
import sys

from Utils import Constraint


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


class SSVEPformer3(nn.Module):
    # 空间滤波器模块，为每个通道分配不同的权重并融合它们
    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
        '''
        block = []
        block.append(Constraint.Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                                     max_norm=1.0))  # (30, 16, 1, 256)
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.GELU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    # 增强模块，使用CNN块吸收数据并输出其稳定特征
    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block,build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride)))  # input(30, 16, 8, 256) output(30, 32, 8, 124)
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.GELU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer
    def __init__(self, depth, attention_kernal_length, chs_num, class_num, dropout):
        super().__init__()
        # token_num = chs_num * 2
        self.F = [chs_num * 1] + [chs_num * 2]
        token_dim = 560

        self.K = 10
        self.S = 2
        output_dim = int((token_dim  - 1 * (self.K - 1) - 1) / self.S + 1)

        net = []
        net.append(self.spatial_block(chs_num, dropout))  # （30， 16， 1， 256）
        net.append(self.enhanced_block(self.F[1], self.F[1], dropout,
                                       self.K, self.S))  # (30, 32, 1, 124)

        self.conv_layers = nn.Sequential(*net)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(self.F[1], self.F[1], 1, padding=1 // 2, groups=1),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.transformer = Transformer(depth, self.F[1], output_dim, attention_kernal_length, dropout)

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * self.F[1], class_num * 6),
            nn.LayerNorm(class_num * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(class_num * 6, class_num)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # input(30, 8, 560)
        x = self.conv_layers(x) # x:(30, 32, 1, 124)
        x = x.squeeze(2)  # (30, 32, 124)
        # x = self.to_patch_embedding(x) # x:(30, 32, 124)
        x = self.transformer(x) # x:(30, 32, 124)
        return self.mlp_head(x)