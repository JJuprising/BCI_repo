import numpy as np
import torch
from fightingcv_attention.attention.CBAM import CBAMBlock
import math
from torch.fft import fft
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
from fightingcv_attention.attention.ECAAttention import ECAAttention

from torch.nn import init
from scipy import signal
from etc.global_config import config
import torch.nn.functional as F
devices = "cuda" if torch.cuda.is_available() else "cpu"
ws = config["data_param_12"]["ws"]
Fs = config["data_param_12"]["Fs"]

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y
# 自注意力
from Utils import Constraint
# 绝对位置编码
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(AbsolutePositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.embedding(pos)
        return x + pos_emb

# 相对位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (100, 512) 1. 创建位置编码张量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (100, 1)2. 生成位置信息
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (256, )3. 计算频率信息
        pe[:, 0::2] = torch.sin(position * div_term)  # 4. 填充偶数列（正弦）
        pe[:, 1::2] = torch.cos(position * div_term)  # 5. 填充奇数列（余弦）
        # pe = pe.unsqueeze(0).transpose(0, 1) # (100, 1, 512)6. 调整形状
        #pe.requires_grad = False
        self.register_buffer('pe', pe)  # 7. 注册为 buffer

    def forward(self, x):
        k = self.pe.repeat(x.size(0), 1, 1)
        return x + k  # 8. 将位置编码添加到输入中

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

        # 点乘操作
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # dots:(30, 8, 16, 16)

        attn = self.attend(dots)  # attn:(30, 8, 16, 16)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # out:(30, 8, 16, 8)
        out = rearrange(out, 'b h n d -> b n (h d)')  # out:(30, 16, 64)
        return self.to_out(out)  # out:(30, 16, 220)


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3) # (30, 8, 256, 8)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1) # (30, 8, 8, 560)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3) # (30, 8, 560, 8)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5  # matmul是矩阵相乘，q1:(30, 8, 256, 8) k2:(30, 8, 8, 560) attn:(30, 8, 256, 560)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output) # (30, 256, 8)

        return output


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
    real_part = np.real(fft_result[:, :, fft_index_start:fft_index_end - 1])
    imag_part = np.imag(fft_result[:, :, fft_index_start:fft_index_end - 1])
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

class convResidual(nn.Module):
    def __init__(self, token_num, token_length, dropout=0.5):
        super().__init__()
        # 对时间序列进行通道融合

        # 使用nn.Sequential封装卷积层
        self.conv_block = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=15, padding=15 // 2, groups=1),
            nn.LayerNorm(token_length),  # 使用层正则化
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(token_num, token_num, kernel_size=9, padding=9 // 2, groups=1),
            nn.LayerNorm(token_length),  # 使用层正则化
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(token_num, token_num, kernel_size=5, padding=5 // 2, groups=1),
            nn.LayerNorm(token_length),  # 使用层正则化
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 残差分支
        residual = x

        # 卷积层块
        x = self.conv_block(x)

        # 残差连接
        x = x + residual

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
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # 将 x 转换为 FloatTensor
        x = x.float()
        x = self.to_patch_embedding(x)
        x = self.transformer(x)
        return x

class TFformer3(Module):
    '''
    T: 时间序列长度
    '''

    def __init__(self, T, depth, heads, chs_num, class_num, tt_dropout, ff_dropout, dim_thead=8, dim_fhead=8, dim=220):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 时间序列的网络层
        time_channel = chs_num * 3
        self.T = T
        self.channels_combination = nn.Sequential(
            nn.Conv1d(chs_num, time_channel , 1, padding=1 // 2, groups=1),
            nn.LayerNorm(T),
            nn.GELU(),
            nn.Dropout(ff_dropout)
        ).to(device)

        # self.convRes = convResidual(token_num=time_channel, token_length=T, dropout=ff_dropout).to(device)
        self.attentionEncoder = ModuleList([])
        self.dropout_level = 0.5
        # self.positional_encoder = PositionalEncoding(chs_num, T)
        # self.positional_encoder = AbsolutePositionalEncoding(chs_num, T)
        for _ in range(depth):
            self.attentionEncoder.append(ModuleList([
                # ECAAttention(kernel_size=3),
                # SimplifiedScaledDotProductAttention(d_model=dim, h=8),
                Attention(time_channel, dim_head=dim_thead, heads=heads, dropout=tt_dropout),
                # Attention(token_num=self.F[0], token_length=self.F[1]),
                nn.LayerNorm(time_channel),
                FeedForward(time_channel, hidden_dim=dim_fhead, dropout=ff_dropout),
                nn.LayerNorm(time_channel)
            ]))
            self.attentionEncoder.to(device)


        # SSVEPformer
        self.subnetwork = SSVEPformer(depth=depth, attention_kernal_length=31, chs_num=chs_num, class_num=class_num,
                                      dropout=ff_dropout)
        self.subnetwork.to(device)


        # 交叉注意力机制
        self.crossAttentionEncoder = ModuleList([])
        for _ in range(1):
            self.crossAttentionEncoder.append(ModuleList([
                CrossAttention(chs_num*2, time_channel , 8, 8, 8),
                nn.LayerNorm(chs_num*2),
                FeedForward(chs_num*2, hidden_dim=16, dropout=ff_dropout),
                nn.LayerNorm(chs_num*2)
            ]).to(device))

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_level),
            nn.Linear(560 * chs_num*2, class_num * 6),
            nn.LayerNorm(class_num * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(class_num * 6, class_num)
        ).to(device)

    # 有两个子网络，一个子网络处理时间序列，一个子网络处理频谱序列
    def forward(self, x):
        # 处理时间序列
        x_t = x
        x_t = x_t
        # print("T shape:", self.T)
        # print("x_t shape:", x_t.shape)
        x_t = self.channels_combination(x_t)
        # x_t = self.convRes(x_t)
        x_t = rearrange(x_t, 'b c t -> b t c')
        # 使用位置编码
        # x_t = self.positional_encoder(x_t)
        for attn, attn_post_norm, ff, ff_post_norm in self.attentionEncoder:
            x_t = attn(x_t) + x_t
            x_t = attn_post_norm(x_t)
            x_t = ff(x_t) + x_t
            x_t = ff_post_norm(x_t) # (30, 256, 8)
        # x_t = rearrange(x_t, 'b t c -> b c t')
        # x_t = self.time_head(x_t) # (30, T // 8)

        # 处理频谱序列
        x_fft = complex_spectrum_features(x, FFT_PARAMS=[Fs, ws])  # x:(30,1,8,256) x_fft:(30, 1, 8, 560)

        # device = torch.device("cuda:0")
        x_fft = torch.tensor(x_fft, dtype=torch.float)
        x_fft = x_fft.to(devices)
        x_fft = self.subnetwork(x_fft)  # (30, T // 8)
        x_fft = rearrange(x_fft, 'b c f -> b f c') # (30, 560, 16)

        # 融合两个子网络的结果
        output = x_fft
        # print("x_t shape:", x_t.shape)
        # print("x_fft shape:", x_fft.shape)
        for attn, attn_post_norm, ff, ff_post_norm in self.crossAttentionEncoder:
            output = attn(output, x_t) + output
            output = attn_post_norm(output)
            output = ff(output) + output
            output = ff_post_norm(output)

        output = self.mlp_head(output)
        return output

class FBTFformer3(Module):
    def __init__(self, T, depth, heads, chs_num, class_num, tt_dropout, ff_dropout, dim_thead=8, dim_fhead=8, dim=220, num_subbands = 3):
        super().__init__()
        self.num_subbands = num_subbands
        # 创建多个 TFformer3 子网络
        self.subnetworks = nn.ModuleList([
            TFformer3(T, depth, heads, chs_num, class_num, tt_dropout, ff_dropout, dim_thead, dim_fhead, dim)
            for _ in range(num_subbands)
        ])
        # 结果融合层
        self.fusion_layer = nn.Conv1d(3, 1, kernel_size=1)

    def forward(self, x):
        # 每个子网络处理对应的子带数据
        subband_outputs = [
            subnetwork(x[:, :, i, :])
            for i, subnetwork in enumerate(self.subnetworks)
        ]

        # 将子网络输出进行融合
        outputs = torch.stack(subband_outputs, dim=2)  # (30, 12, 3)
        outputs = torch.transpose(outputs, 1, 2)  # (30, 3, 12)
        fused_output = self.fusion_layer(outputs)  # (30, 1, 12)
        return fused_output.squeeze(dim=1)
