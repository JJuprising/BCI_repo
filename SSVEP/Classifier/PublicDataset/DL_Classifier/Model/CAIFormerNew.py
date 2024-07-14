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

ws = config["data_param_12"]["ws"]
Fs = config["data_param_12"]["Fs"]
#自注意力
from Utils import Constraint

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
        qkv = self.to_qkv(x).chunk(3, dim=-1) # x:(30, 16, 220) qkv:(tuple:3)[0:(30, 16, 64),1:(30, 16, 64),2:(30, 16, 64)]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv) # q:(30, 8, 16, 8) (batch_size, channels, heads * dim_head)

        # 点乘操作
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # dots:(30, 8, 16, 16)

        attn = self.attend(dots)  # attn:(30, 8, 16, 16)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) # out:(30, 8, 16, 8)
        out = rearrange(out, 'b h n d -> b n (h d)') # out:(30, 16, 64)
        return self.to_out(out) # out:(30, 16, 220)
# feedforward
from Utils.Constraint import Conv2dWithConstraint

def complex_spectrum_features(segmented_data, FFT_PARAMS):
    sample_freq = FFT_PARAMS[0]
    time_len = FFT_PARAMS[1]
    resolution, start_freq, end_freq = 0.2930, 3, 35
    NFFT = round(sample_freq / resolution)
    fft_index_start = int(round(start_freq / resolution))
    fft_index_end = int(round(end_freq / resolution)) + 1
    sample_point = int(sample_freq * time_len)
    fft_result = np.fft.fft(segmented_data, axis=-1, n=NFFT) / (sample_point / 2)
    real_part = np.real(fft_result[:, :, :, fft_index_start:fft_index_end])
    imag_part = np.imag(fft_result[:, :, :, fft_index_start:fft_index_end])
    features_data = np.concatenate([real_part, imag_part], axis=-1)
    return features_data


# class Attention(nn.Module):
#     def __init__(self, token_num, token_length, kernal_length=31, dropout=0.5):
#         super().__init__()
#
#         self.att2conv = nn.Sequential(
#             nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2, groups=1),
#             nn.LayerNorm(token_length),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )
#
#     def forward(self, x):
#         out = self.att2conv(x)
#         return out





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

# main class
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
class iTransformerFFT(Module):
    def __init__(self,T, depth, heads,chs_num, class_num, tt_dropout,ff_dropout,dim_thead=8,dim_fhead=8,dim=220):
        super().__init__()
        self.layers = ModuleList([])
        self.dropout_level = 0.5
        self.F = [chs_num * 2] + [chs_num * 4]
        self.K = 10
        self.S = 2
        self.F1 = 220
        self.D = 1
        self.F2 = self.F1 * self.D
        self.kernelength1 = T
        self.kernelength2 = T // 16


        #dim = int(fs*Ws)

        self.mlp_in = nn.Sequential(
            nn.Linear(dim, dim *chs_num),
            Rearrange('b v (n d) -> b (v n) d', n = chs_num),
            nn.LayerNorm(dim)
        )

        self.fft_mlp_in = nn.Sequential(
            Rearrange('b v n c -> b v (n c)'),
            nn.Linear(dim * 2, dim * chs_num),
            Rearrange('b v (n d) -> b (v n) d', n = chs_num),
            nn.LayerNorm(dim)
        )


        self.fc = nn.Linear(220, class_num)
        self.conv1d = nn.Conv1d(128,64, 3, 1)
        self.pool = nn.MaxPool1d(2)
        self.mlp= MLPClassifier( 128 * 256  , 256, 12)





        net = []
        # 这一段代码的主要目的应该是为了要实现数据集维度的转换
        # 输入维度的8和输出维度的8经过这段处理后含义已经不一样了
        # 结合后面的代码来看，这会误解8的含义，使得神经网络的含义难以解释
        net.append(nn.Conv2d(1, self.F1, (1, self.kernelength1), bias=False, padding="same"))# input(30, 1, 8, 256) output(30, F1, 8, 256)
        net.append(nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)) # (30, F1, 8, 256)
        net.append(Conv2dWithConstraint(self.F1, self.F1 * self.D, (chs_num ,1), max_norm=1,groups=self.F1, bias=False)) # [30, F1*D, 1, 256)
        net.append(nn.BatchNorm2d(self.D * self.F1, momentum=0.01, affine=True, eps=1e-3)) # [30, F1*D, 1, 256)
        net.append(nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))) # (30, F1*D, 1, 64)
        #pointwiseConv2
        net.append(nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelength2),
                                        groups=self.F1 * self.D,
                                        bias=False, padding="same")) # # (30, F1*D, 1, 64)
        #depthwiseConv
        net.append(nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False)) # (30, F2, 1, 64)
        net.append(nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)) # (30, F2, 1, 64)
        net.append(nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))) # (30, F2, 1, 8)
        self.conv_layers = nn.Sequential(*net)



        for _ in range(depth):
            self.layers.append(ModuleList([
                # ECAAttention(kernel_size=3),
                # SimplifiedScaledDotProductAttention(d_model=dim, h=8),
                Attention(dim, dim_head = dim_thead, heads = heads, dropout = tt_dropout),
                #Attention(token_num=self.F[0], token_length=self.F[1]),
                nn.LayerNorm(dim),
                FeedForward(dim, hidden_dim=dim_fhead, dropout=ff_dropout),
                nn.LayerNorm(dim)
            ]))

    def forward( self,x):


        #x = x.squeeze(1)
        #x_fft = fft(x)#(30,8,256)
        #x_fft = torch.view_as_real(x_fft)#(30,8,256,2)
        #x = self.mlp_in(x)#(30,64,256)
        #(30,256,124)
        #x_fft = self.fft_mlp_in(x_fft)#(30,64,256)
        #x, fft_ps = pack([x_fft, x], 'b * d')
        #(30,128,256)
        x_fft = complex_spectrum_features(x, FFT_PARAMS=[Fs, ws])# x:(30,1,8,256) x_fft:(30, 1, 8, 220)

        #时间序列
        x = self.conv_layers(x)#（30，220，1，8）
        x = x.permute(0,2,3,1)

        x = torch.Tensor(x.squeeze(1))
        x_fft =torch.Tensor(x_fft.squeeze(1)) # (30, 8, 220) 批次大小，通道数，频率序列

        x = torch.cat([x_fft, x], dim=1) # (30, 16, 220)

        #x, fft_ps = pack([x_fft, x], 'b * d')

        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            x = attn(x) + x
            x = attn_post_norm(x)
            x = ff(x) + x
            x = ff_post_norm(x)
        #（30，128,256）
        #这里可以加一些层
        #x = self.conv1d(x)

        #x = self.pool(x)
        x = x.mean(dim=1) # (30, 220)
        x = self.fc(x) # (30, 12)

        #x = self.mlp(x)
        return x