from scipy import signal

import numpy as np
import torch
from fightingcv_attention.attention.CBAM import CBAMBlock

from torch.fft import fft
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
from fightingcv_attention.attention.ECAAttention import ECAAttention

from torch.nn import init

from etc.global_config import config

devices = "cuda" if torch.cuda.is_available() else "cpu"
ws = config["data_param_12"]["ws"]
Fs = config["data_param_12"]["Fs"]
# 自注意力
from Utils import Constraint


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    x = data.cpu().numpy()  # 先将张量复制到 CPU，然后再转换为 NumPy 数组
    # 应用滤波器
    y = signal.lfilter(b, a, x)

    # 将 y 转换回 PyTorch 张量
    y = torch.from_numpy(y)
    return y


#
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3,
                                   dim=-1)  # x:(30, 16, 220) qkv:(tuple:3)[0:(30, 16, 64),1:(30, 16, 64),2:(30, 16, 64)]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                      qkv)  # q:(30, 8, 16, 8) (batch_size, channels, heads * dim_head)
        qkv = self.to_qkv(x).chunk(3,
                                   dim=-1)  # x:(30, 16, 220) qkv:(tuple:3)[0:(30, 16, 64),1:(30, 16, 64),2:(30, 16, 64)]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                      qkv)  # q:(30, 8, 16, 8) (batch_size, channels, heads * dim_head)

        # 点乘操作
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # dots:(30, 8, 16, 16)

        attn = self.attend(dots)  # attn:(30, 8, 16, 16)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # out:(30, 8, 16, 8)
        out = rearrange(out, 'b h n d -> b n (h d)')  # out:(30, 16, 64)
        return self.to_out(out)  # out:(30, 16, 220)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def complex_spectrum_features(segmented_data, FFT_PARAMS):
    sample_freq = FFT_PARAMS[0]
    time_len = FFT_PARAMS[1]
    resolution, start_freq, end_freq = 0.2, 8, 64
    NFFT = round(sample_freq / resolution)
    fft_index_start = int(round(start_freq / resolution))
    fft_index_end = int(round(end_freq / resolution)) + 1
    sample_point = int(sample_freq * time_len)
    # 将 segmented_data 移动到 CPU
    segmented_data_cpu = segmented_data.cpu()

    # 转换为 NumPy 数组
    segmented_data_np = segmented_data_cpu.numpy()

    # 进行 FFT
    fft_result = np.fft.fft(segmented_data_np, axis=-1, n=NFFT) / (sample_point / 2)
    real_part = np.real(fft_result[:, :, :, fft_index_start:fft_index_end - 1])
    imag_part = np.imag(fft_result[:, :, :, fft_index_start:fft_index_end - 1])
    features_data = np.concatenate([real_part, imag_part], axis=-1)
    return features_data


class convAttention(nn.Module):
    def __init__(self, token_num, token_length, kernal_length=31, dropout=0.5):
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


class freqFeedForward(nn.Module):
    def __init__(self, token_length, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_length, token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, token_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class convTransformer(nn.Module):
    def __init__(self, depth, token_num, token_length, kernal_length, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(token_length, convAttention(token_num, token_length, kernal_length, dropout=dropout)),
                PreNorm(token_length, freqFeedForward(token_length, dropout=dropout))
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

        self.transformer = convTransformer(depth, token_num, token_dim, attention_kernal_length, dropout)

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
        # 将 x 转换为 FloatTensor
        x = x.float()  # (30,8,256)
        x = self.to_patch_embedding(x)
        x = self.transformer(x)
        return self.mlp_head(x)


class TFFBformer(Module):
    '''
    T: 时间序列长度
    '''

    def __init__(self, T, depth, heads, chs_num, class_num, tt_dropout, ff_dropout, dim_thead=8, dim_fhead=8, dim=220):
        super().__init__()
        # 时间序列的网络层
        self.attentionEncoder = ModuleList([])
        self.fc = nn.Linear(256, class_num)
        self.Linear = nn.Linear(T, dim)
        self.T = T
        self.dropout_level = 0.5
        self.F = [chs_num * 2] + [chs_num * 4]
        self.K = 10
        self.S = 2
        self.cbam_block1 = CBAMBlock(channel=self.F[1], reduction=16, kernel_size=7)
        for _ in range(depth):
            self.attentionEncoder.append(ModuleList([
                # ECAAttention(kernel_size=3),
                # SimplifiedScaledDotProductAttention(d_model=dim, h=8),
                Attention(dim, dim_head=dim_thead, heads=heads, dropout=tt_dropout),
                # Attention(token_num=self.F[0], token_length=self.F[1]),
                nn.LayerNorm(dim),
                FeedForward(dim, hidden_dim=dim_fhead, dropout=ff_dropout),
                nn.LayerNorm(dim)
            ]))
        net = []
        net.append(self.spatial_block(chs_num, self.dropout_level))
        net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level,
                                       1, 1))

        self.conv_layers = nn.Sequential(*net)

        # SSVEPformer
        self.subnetwork = SSVEPformer(depth=depth, attention_kernal_length=31, chs_num=chs_num, class_num=class_num,
                                      dropout=ff_dropout)
        self.subnetworks = nn.ModuleList([
            SSVEPformer(depth, 31, chs_num, class_num, ff_dropout)
            for _ in range(3)
        ])

        # 结果融合层
        self.fusion_layer = nn.Conv1d(3, 1, kernel_size=1)

    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
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
           Enhanced structure block,build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride)))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    # 有两个子网络，一个子网络处理时间序列，一个子网络处理频谱序列
    def forward(self, x):
        # 处理时间序列
        # 通过线性层对序列进行裁剪
        # x_t = self.Linear(x) # (30, 1, 8, 220)
        # x_t = torch.Tensor(x_t.squeeze(1))
        #
        # for attn, attn_post_norm, ff, ff_post_norm in self.attentionEncoder:
        #     x_t = attn(x_t) + x_t
        #     x_t = attn_post_norm(x_t)
        #     x_t = ff(x_t) + x_t
        #     x_t = ff_post_norm(x_t)

        x_t = self.conv_layers(x)
        x_t = x_t.squeeze(2)

        x_t = x_t.mean(dim=1)  # (30, 220)

        x_t = self.fc(x_t)  # (30, 12)

        # 处理频谱序列
        subband_data = []
        for i in range(3):
            lowcut = i * 8 + 2  # 根据数据集 1 的特性设置截止频率
            highcut = 80
            filtered_data = butter_bandpass_filter(x, lowcut, highcut, self.T)
            filtered_data = complex_spectrum_features(filtered_data,
                                                      FFT_PARAMS=[Fs, ws])  # 数据维度：(36, 1, 8, 560)
            filtered_data = torch.from_numpy(filtered_data)
            subband_data.append(filtered_data)
        # 需要转换维度为(36, 8, 3, 256)
        subband_data = torch.stack(subband_data, dim=3)
        subband_data = subband_data.squeeze(1)
        subband_data = subband_data.to(x.device)
        subband_outputs = [
            subnetwork(subband_data[:, :, i, :])  # (30, 12)
            for i, subnetwork in enumerate(self.subnetworks)
        ]
        # 将子网络输出进行融合
        outputs = torch.stack(subband_outputs, dim=2)  # (30, 12, 3)
        outputs = torch.transpose(outputs, 1, 2)  # (30, 3, 12)
        fused_output = self.fusion_layer(outputs)  # (30, 1, 12)
        x_fft = fused_output.squeeze(1)

        # 计算 x_t 和 x_fft 在第二维上的大小

        weight_x_t = 0.4
        weight_x_fft = 0.6

        # 分别应用权重
        weighted_x_t = torch.mul(x_t, weight_x_t)
        weighted_x_fft = torch.mul(x_fft, weight_x_fft)

        # 加权求和
        fused_output = weighted_x_t + weighted_x_fft
        return fused_output