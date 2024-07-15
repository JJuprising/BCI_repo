import numpy as np
import torch
from fightingcv_attention.attention.CBAM import CBAMBlock

from torch.fft import fft
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
from fightingcv_attention.attention.ECAAttention import ECAAttention
import torch.nn.functional as F
import math
from torch.nn import init

from etc.global_config import config
import torch.nn.functional as F
# 自注意力
from Utils import Constraint
devices = "cuda" if torch.cuda.is_available() else "cpu"
classes = config['classes'] # 数据集
if classes == 12:
    ws = config["data_param_12"]["ws"]
    Fs = config["data_param_12"]["Fs"]
elif classes == 40:
    ws = config["data_param_40"]["ws"]
    Fs = config["data_param_40"]["Fs"]


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
        # q, k = RoPE(q, k)
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
    def __init__(self, out_dim, depth, attention_kernal_length, chs_num, class_num, dropout):
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
            nn.Linear(token_dim * token_num, out_dim[0]),
            nn.LayerNorm(out_dim[0]),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(out_dim[0], out_dim[1])
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # 将 x 转换为 FloatTensor
        x = x.float()
        x = self.to_patch_embedding(x)
        x = self.transformer(x)
        # return self.mlp_head(x)
        return x

class TFformer3(Module):
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
    '''
    T: 时间序列长度
    '''

    def __init__(self, T, depth, heads, chs_num, class_num, tt_dropout, ff_dropout, dim_thead=8, dim_fhead=8, dim=220):
        super().__init__()
        # 时间序列的网络层
        self.attentionEncoder = ModuleList([])
        self.dropout_level = 0.5
        self.fc = nn.Linear(int(Fs*ws), class_num)
        self.F = [chs_num * 2] + [chs_num * 1]
        for _ in range(depth):
            self.attentionEncoder.append(ModuleList([
                # ECAAttention(kernel_size=3),
                # SimplifiedScaledDotProductAttention(d_model=dim, h=8),
                Attention(chs_num, dim_head=dim_thead, heads=heads, dropout=tt_dropout),
                # Attention(token_num=self.F[0], token_length=self.F[1]),
                nn.LayerNorm(chs_num),
                FeedForward(chs_num, hidden_dim=dim_fhead, dropout=ff_dropout),
                nn.LayerNorm(chs_num)
            ]))



        self.out_dim = [class_num * 4, class_num * 2]
        self.time_conv_kernel = 7
        self.time_conv_dim = T - self.time_conv_kernel + 1
        self.time_head = nn.Sequential(
            nn.Conv1d(chs_num, self.out_dim[0], self.time_conv_kernel),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),# 平均池化，降维
            nn.Flatten(),
            nn.Linear(self.out_dim[0], self.out_dim[1]),
            # nn.LayerNorm(self.out_dim[1]),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(self.out_dim[1], class_num),
            nn.GELU()
        )
        net = []
        net.append(self.spatial_block(chs_num, self.dropout_level))
        net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level,
                                  1, 1))

        self.conv_layers = nn.Sequential(*net)
        # SSVEPformer
        self.subnetwork = SSVEPformer(out_dim = self.out_dim, depth=depth, attention_kernal_length=31, chs_num=chs_num, class_num=class_num,
                                      dropout=ff_dropout)

        # 结果融合层
        # self.fusion_layer = nn.Conv1d(2, 1, kernel_size=1)

        # 全连接层融合
        self.fully_connected = nn.Sequential(
            nn.Linear(self.out_dim[1] + class_num, class_num),
            nn.LayerNorm(class_num),
            nn.GELU()
        )

        # 交叉注意力机制
        self.crossAttentionEncoder = ModuleList([])
        for _ in range(depth):
            self.crossAttentionEncoder.append(ModuleList([
                CrossAttention(chs_num*2, chs_num, 8, 8, 8),
                nn.LayerNorm(chs_num*2),
                FeedForward(chs_num*2, hidden_dim=16, dropout=ff_dropout),
                nn.LayerNorm(chs_num*2)
            ]))

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_level),
            nn.Linear(560 * chs_num*2, class_num * 6),
            # nn.Linear(77 * chs_num*2, class_num * 6),# x_t
            nn.LayerNorm(class_num * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(class_num * 6, class_num)
        )

    # 有两个子网络，一个子网络处理时间序列，一个子网络处理频谱序列
    def forward(self, x):
        # 处理时间序列
        x_t = x
        x_t = x_t.squeeze(1)
        x_t = rearrange(x_t, 'b c t -> b t c')
        for attn, attn_post_norm, ff, ff_post_norm in self.attentionEncoder:
            x_t = attn(x_t) + x_t
            x_t = attn_post_norm(x_t)
            x_t = ff(x_t) + x_t
            x_t = ff_post_norm(x_t) # (30, 256, 8)
   

        # 处理频谱序列
        x_fft = complex_spectrum_features(x, FFT_PARAMS=[Fs, ws])  # x:(30,1,8,256) x_fft:(30, 1, 8, 560)
        x_fft = torch.tensor(x_fft.squeeze(1), dtype=torch.float)
        x_fft = x_fft.to(devices)
        x_fft = self.subnetwork(x_fft)  # (30, T // 8)
        x_fft = rearrange(x_fft, 'b c f -> b f c') # (30, 560, 16)

        # print(x_t.shape)# torch.Size([64, 250, 9])
        # print(x_fft.shape)# torch.Size([64, 560, 18])

        # 融合两个子网络的结果
        for attn, attn_post_norm, ff, ff_post_norm in self.crossAttentionEncoder:
            output = attn(x_fft, x_t) + x_fft
            output = attn_post_norm(output)
            output = ff(output) + output
            output = ff_post_norm(output)
        output = self.mlp_head(output)
        # output = self.mlp_head(x_t)
        return output

# import numpy as np
# import torch
# from fightingcv_attention.attention.CBAM import CBAMBlock

# from torch.fft import fft
# from torch import nn, einsum, Tensor
# from torch.nn import Module, ModuleList
# from einops import rearrange, reduce, repeat, pack, unpack
# from einops.layers.torch import Rearrange
# from fightingcv_attention.attention.ECAAttention import ECAAttention
# import torch.nn.functional as F
# import math
# from torch.nn import init

# from etc.global_config import config
# import torch.nn.functional as F
# # 自注意力
# from Utils import Constraint
# devices = "cuda" if torch.cuda.is_available() else "cpu"
# classes = config['classes'] # 数据集
# if classes == 12:
#     ws = config["data_param_12"]["ws"]
#     Fs = config["data_param_12"]["Fs"]
# elif classes == 40:
#     ws = config["data_param_40"]["ws"]
#     Fs = config["data_param_40"]["Fs"]

# def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
#     # (max_len, 1)
#     position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
#     # (output_dim//2)
#     ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
#     theta = torch.pow(10000, -2 * ids / output_dim)

#     # (max_len, output_dim//2)
#     embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))

#     # (max_len, output_dim//2, 2)
#     embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

#     # (bs, head, max_len, output_dim//2, 2)
#     embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复

#     # (bs, head, max_len, output_dim)
#     # reshape后就是：偶数sin, 奇数cos了
#     embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
#     embeddings = embeddings.to(device)
#     return embeddings


# # %%

# def RoPE(q, k):
#     # q,k: (bs, head, max_len, output_dim)
#     batch_size = q.shape[0]
#     nums_head = q.shape[1]
#     max_len = q.shape[2]
#     output_dim = q.shape[-1]

#     # (bs, head, max_len, output_dim)
#     pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

#     # cos_pos,sin_pos: (bs, head, max_len, output_dim)
#     # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
#     cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
#     sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

#     # q,k: (bs, head, max_len, output_dim)
#     q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
#     q2 = q2.reshape(q.shape)  # reshape后就是正负交替了

#     # 更新qw, *对应位置相乘
#     q = q * cos_pos + q2 * sin_pos

#     k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
#     k2 = k2.reshape(k.shape)
#     # 更新kw, *对应位置相乘
#     k = k * cos_pos + k2 * sin_pos

#     return q, k

# #
# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3,
#                                    dim=-1)  # x:(30, 16, 220) qkv:(tuple:3)[0:(30, 16, 64),1:(30, 16, 64),2:(30, 16, 64)]
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
#                       qkv)  # q:(30, 8, 16, 8) (batch_size, channels, heads * dim_head)
#         q, k = RoPE(q, k)
#         # 点乘操作
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # dots:(30, 8, 16, 16)

#         attn = self.attend(dots)  # attn:(30, 8, 16, 16)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)  # out:(30, 8, 16, 8)
#         out = rearrange(out, 'b h n d -> b n (h d)')  # out:(30, 16, 64)
#         return self.to_out(out)  # out:(30, 16, 220)


# class CrossAttention(nn.Module):
#     def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.k_dim = k_dim
#         self.v_dim = v_dim

#         self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
#         self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
#         self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
#         self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

#     def forward(self, x1, x2, mask=None):
#         batch_size, seq_len1, in_dim1 = x1.size()
#         seq_len2 = x2.size(1)

#         q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3) # (30, 8, 256, 8)
#         k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1) # (30, 8, 8, 560)
#         v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3) # (30, 8, 560, 8)



#         attn = torch.matmul(q1, k2) / self.k_dim ** 0.5  # matmul是矩阵相乘，q1:(30, 8, 256, 8) k2:(30, 8, 8, 560) attn:(30, 8, 256, 560)

#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e9)

#         attn = F.softmax(attn, dim=-1)
#         output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
#         output = self.proj_o(output) # (30, 256, 8)

#         return output


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.net(x)


# def complex_spectrum_features(segmented_data, FFT_PARAMS):
#     sample_freq = FFT_PARAMS[0]
#     time_len = FFT_PARAMS[1]
#     resolution, start_freq, end_freq = 0.2, 8, 64
#     NFFT = round(sample_freq / resolution)
#     fft_index_start = int(round(start_freq / resolution))
#     fft_index_end = int(round(end_freq / resolution)) + 1
#     sample_point = int(sample_freq * time_len)
#     # 将 segmented_data 移动到 CPU
#     segmented_data_cpu = segmented_data.cpu()

#     # 转换为 NumPy 数组
#     segmented_data_np = segmented_data_cpu.numpy()

#     # 进行 FFT
#     fft_result = np.fft.fft(segmented_data_np, axis=-1, n=NFFT) / (sample_point / 2)
#     real_part = np.real(fft_result[:, :, :, fft_index_start:fft_index_end - 1])
#     imag_part = np.imag(fft_result[:, :, :, fft_index_start:fft_index_end - 1])
#     features_data = np.concatenate([real_part, imag_part], axis=-1)
#     return features_data


# class convAttention(nn.Module):
#     def __init__(self, token_num, token_length, kernal_length=31, dropout=0.5):
#         super().__init__()

#         self.att2conv = nn.Sequential(
#             nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2, groups=1),
#             nn.LayerNorm(token_length),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x):
#         out = self.att2conv(x)
#         return out


# class freqFeedForward(nn.Module):
#     def __init__(self, token_length, dropout):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(token_length, token_length),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x):
#         return self.net(x)


# class PreNorm(nn.Module):
#     def __init__(self, token_dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(token_dim)
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)


# class convTransformer(nn.Module):
#     def __init__(self, depth, token_num, token_length, kernal_length, dropout):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(token_length, convAttention(token_num, token_length, kernal_length, dropout=dropout)),
#                 PreNorm(token_length, freqFeedForward(token_length, dropout=dropout))
#             ]))

#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x


# class SSVEPformer(nn.Module):
#     def __init__(self, out_dim, depth, attention_kernal_length, chs_num, class_num, dropout):
#         super().__init__()
#         token_num = chs_num * 2
#         token_dim = 560
#         self.to_patch_embedding = nn.Sequential(
#             nn.Conv1d(chs_num, token_num, 1, padding=1 // 2, groups=1),
#             nn.LayerNorm(token_dim),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )

#         self.transformer = convTransformer(depth, token_num, token_dim, attention_kernal_length, dropout)

#         self.mlp_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(dropout),
#             nn.Linear(token_dim * token_num, out_dim[0]),
#             nn.LayerNorm(out_dim[0]),
#             nn.GELU(),
#             nn.Dropout(0.5),
#             nn.Linear(out_dim[0], out_dim[1])
#         )

#         for m in self.modules():
#             if isinstance(m, (nn.Conv1d, nn.Linear)):
#                 nn.init.normal_(m.weight, mean=0.0, std=0.01)

#     def forward(self, x):
#         # 将 x 转换为 FloatTensor
#         x = x.float()
#         x = self.to_patch_embedding(x)
#         x = self.transformer(x)
#         # return self.mlp_head(x)
#         return x

# class TFformer3(Module):
#     def spatial_block(self, nChan, dropout_level):
#         '''
#            Spatial filter block,assign different weight to different channels and fuse them
#         '''
#         block = []
#         block.append(Constraint.Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
#                                                      max_norm=1.0))
#         block.append(nn.BatchNorm2d(num_features=nChan * 2))
#         block.append(nn.PReLU())
#         block.append(nn.Dropout(dropout_level))
#         layer = nn.Sequential(*block)
#         return layer

#     def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
#         '''
#            Enhanced structure block,build a CNN block to absorb data and output its stable feature
#         '''
#         block = []
#         block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
#                                stride=(1, stride)))
#         block.append(nn.BatchNorm2d(num_features=out_channels))
#         block.append(nn.PReLU())
#         block.append(nn.Dropout(dropout_level))
#         layer = nn.Sequential(*block)
#         return layer
#     '''
#     T: 时间序列长度
#     '''

#     def __init__(self, T, depth, heads, chs_num, class_num, tt_dropout, ff_dropout, dim_thead=8, dim_fhead=8, dim=220):
#         super().__init__()
#         # 时间序列的网络层
#         self.attentionEncoder = ModuleList([])
#         self.dropout_level = 0.5
#         self.fc = nn.Linear(int(Fs*ws), class_num)
#         self.F = [chs_num * 2] + [chs_num * 1]
#         for _ in range(depth):
#             self.attentionEncoder.append(ModuleList([
#                 # ECAAttention(kernel_size=3),
#                 # SimplifiedScaledDotProductAttention(d_model=dim, h=8),
#                 Attention(chs_num, dim_head=dim_thead, heads=heads, dropout=tt_dropout),
#                 # Attention(token_num=self.F[0], token_length=self.F[1]),
#                 nn.LayerNorm(chs_num),
#                 FeedForward(chs_num, hidden_dim=dim_fhead, dropout=ff_dropout),
#                 nn.LayerNorm(chs_num)
#             ]))



#         self.out_dim = [class_num * 4, class_num * 2]
#         self.time_conv_kernel = 7
#         self.time_conv_dim = T - self.time_conv_kernel + 1
#         self.time_head = nn.Sequential(
#             nn.Conv1d(chs_num, self.out_dim[0], self.time_conv_kernel),
#             nn.GELU(),
#             nn.AdaptiveAvgPool1d(1),# 平均池化，降维
#             nn.Flatten(),
#             nn.Linear(self.out_dim[0], self.out_dim[1]),
#             # nn.LayerNorm(self.out_dim[1]),
#             nn.Sigmoid(),
#             nn.Dropout(0.5),
#             nn.Linear(self.out_dim[1], class_num),
#             nn.GELU()
#         )
#         net = []
#         net.append(self.spatial_block(chs_num, self.dropout_level))
#         net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level,
#                                   1, 1))

#         self.conv_layers = nn.Sequential(*net)
#         # SSVEPformer
#         self.subnetwork = SSVEPformer(out_dim = self.out_dim, depth=depth, attention_kernal_length=31, chs_num=chs_num, class_num=class_num,
#                                       dropout=ff_dropout)

#         # 结果融合层
#         # self.fusion_layer = nn.Conv1d(2, 1, kernel_size=1)

#         # 全连接层融合
#         self.fully_connected = nn.Sequential(
#             nn.Linear(self.out_dim[1] + class_num, class_num),
#             nn.LayerNorm(class_num),
#             nn.GELU()
#         )

#         # 交叉注意力机制
#         self.crossAttentionEncoder = ModuleList([])
#         for _ in range(depth):
#             self.crossAttentionEncoder.append(ModuleList([
#                 CrossAttention(chs_num*2, chs_num, 8, 8, 8),
#                 nn.LayerNorm(chs_num*2),
#                 FeedForward(chs_num*2, hidden_dim=16, dropout=ff_dropout),
#                 nn.LayerNorm(chs_num*2)
#             ]))

#         self.mlp_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(self.dropout_level),
#             nn.Linear(560 * chs_num * 2, class_num * 6),
#             # nn.Linear(11408, class_num * 6),
#             nn.LayerNorm(class_num * 6),
#             nn.GELU(),
#             nn.Dropout(0.5),
#             nn.Linear(class_num * 6, class_num)
#         )

#     # 有两个子网络，一个子网络处理时间序列，一个子网络处理频谱序列
#     def forward(self, x):
#         # 处理时间序列
#         x_t = x
#         x_t = x_t.squeeze(1)
#         x_t = rearrange(x_t, 'b c t -> b t c')
#         for attn, attn_post_norm, ff, ff_post_norm in self.attentionEncoder:
#             x_t = attn(x_t) + x_t
#             x_t = attn_post_norm(x_t)
#             x_t = ff(x_t) + x_t
#             x_t = ff_post_norm(x_t) # (30, 256, 8)
#         print("x_t.shape",x_t.shape) # x_t.shape torch.Size([64, 153, 8])

#         # 处理频谱序列
#         x_fft = complex_spectrum_features(x, FFT_PARAMS=[Fs, ws])  # x:(30,1,8,256) x_fft:(30, 1, 8, 560)
#         # device = torch.device("cuda:0")
#         x_fft = torch.tensor(x_fft.squeeze(1), dtype=torch.float)
#         x_fft = x_fft.to(devices)
#         x_fft = self.subnetwork(x_fft)  # (30, T // 8)
#         x_fft = rearrange(x_fft, 'b c f -> b f c') # (30, 560, 16)
#         print("x_fft.shape",x_fft.shape) # x_fft.shape torch.Size([64, 560, 16])
#         # 融合两个子网络的结果
#         # for attn, attn_post_norm, ff, ff_post_norm in self.crossAttentionEncoder:
#         #     output = attn(x_fft, x_t) + x_fft
#         #     output = attn_post_norm(output)
#         #     output = ff(output) + output
#         #     output = ff_post_norm(output)
#         # Adjust dimensions for concatenation
#         x_t_transformed = nn.Linear(x_t.shape[-1], x_fft.shape[-1]).to(devices)(x_t)

#         # Concatenate x_t and x_fft along the feature dimension
#         output = torch.cat((x_t_transformed, x_fft), dim=1)
#         print("output.shape:", output.shape) # output.shape: torch.Size([64, 560, 16])
#         output = self.mlp_head(output)
#         print("output.shape:",output.shape)
#         return output