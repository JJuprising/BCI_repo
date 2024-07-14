# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/3/7 19:36
import torch
from torch import nn

'''
L2 范数，也称为欧几里得范数，是向量元素平方和的平方根
自定义卷积层通过对卷积核权重进行 L2 范数约束，实现了权重正则化
'''
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, X):
        '''
        对 self.weight.data 中的每个权重向量 (第 0 维) 计算 L2 范数。
        如果 L2 范数大于 max_norm，则将权重向量缩放到 L2 范数等于 max_norm。
        将修改后的权重张量赋值回 self.weight.data。
        '''
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(X)

def Spectral_Normalization(m):
    for name, layer in m.named_children():
        m.add_module(name, Spectral_Normalization(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()