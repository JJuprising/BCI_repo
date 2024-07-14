import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch
from torch import nn
from Utils import Constraint
import torch.nn.functional as F
import cv2
from fightingcv_attention.attention.CBAM import CBAMBlock

import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2  # 为了保持特征图大小不变，进行边界补零
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        # 第一层级池化层
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        # 第二层级池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三层级池化层
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 特征衔接层
        self.concat_layer = nn.Conv2d(in_channels=input_channels * 3, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        # 第一层级池化
        out1 = self.pool1(x)
        # 第二层级池化
        out2 = self.pool2(x)
        # 第三层级池化
        out3 = self.pool3(x)


def ssvep_to_wavelet_spectrogram(ssvep_batch, sampling_rate, wavelet='morl', levels=6):
    batch_size, num_channels, num_samples = ssvep_batch.shape
    batch_spectrogram_features = np.zeros((batch_size, num_channels, levels, num_samples), dtype=np.float32)

    for i in range(batch_size):
        for j in range(num_channels):
            # 获取当前批次和通道的信号
            ssvep_signal = ssvep_batch[i, j, :]

            # 计算小波系数
            coefficients, frequencies = pywt.cwt(ssvep_signal, np.arange(1, levels + 1), wavelet, sampling_period=1 / sampling_rate)

            # 将时频图特征存储在数组中
            batch_spectrogram_features[i, j, :, :] = np.abs(coefficients)

            #绘制时频图
            # plt.figure(figsize=(10, 6))
            # plt.imshow(np.abs(coefficients), extent=[0, num_samples / sampling_rate, frequencies[-1], frequencies[0]],
            #            aspect='auto', cmap='jet')
            # plt.colorbar(label='Magnitude')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Frequency (Hz)')
            # plt.title('Wavelet Spectrogram')
            # plt.show()

    return batch_spectrogram_features

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_classes)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 将输入张量展平为 (batch_size, 32 * 1 * 124)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ESNet(nn.Module):

    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block, assign different weights to different channels and fuse them
        '''
        block = []
        block.append(Constraint.Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                                     max_norm=1.0))
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))

        layer = nn.Sequential(*block)
        return layer

    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block, build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride)))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))

        layer = nn.Sequential(*block)
        return layer



    def __init__(self, num_channels,  levels):
        super(ESNet, self).__init__()
        self.dropout_level = 0.5
        self.F = [num_channels * 2] + [num_channels * 4]
        self.G = [256,124]
        self.K = 2
        self.S = 2
        # self.spatial_conv = self.spatial_block(num_channels, self.dropout_level)
        # self.enhanced_conv =   self.enhanced_block(self.F[0], self.F[1], self.dropout_level, self.K, self.S)
        # #self.enhanced_conv = self.enhanced_block(256, self.F[1], self.dropout_level, self.K, self.S)
        # self.cbam_block1 = CBAMBlock(channel=self.F[0], reduction=16, kernel_size=7)
        # self.cbam_block2 = CBAMBlock(channel=self.F[1], reduction=16, kernel_size=7)
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1)
        self.relu = nn.PReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv_reduce_channels = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1)

        # self.mlp = nn.Sequential(
        #     nn.Linear(32 * 1 * 124, 512),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 12)
        # )
        self.mlp = MLP(input_size=32*1*64, hidden_size=512, output_classes=12)
        #self.mlp = MLP(input_size=32 * 1 * 21, hidden_size=512, output_classes=12)0.2
        #self.mlp = MLP(input_size=32 * 1 * 60, hidden_size=512, output_classes=12)0.5
        #self.mlp = MLP(input_size=32 * 1 * 508, hidden_size=512, output_classes=12)
        self.block1 = ConvBlock(in_channels=8, out_channels=16, kernel_size=5, stride=2)
        self.block2 =  ConvBlock(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.block3  =  ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        #self.pool = nn.AdaptiveAvgPool1d(256)
        # self.pool  = MultiScaleFeatureFusion(input_channels=64, output_channels=12)
        self.pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_layer = nn.Dropout(p=0.5)

        self.result = nn.Linear(64 * 4 * 4, 12)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.mlp(x)
        return x


