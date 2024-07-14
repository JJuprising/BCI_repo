# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/3/20 13:41
import torch
from torch import nn
from Utils import Constraint
class LSTM(nn.Module):
    '''
        Employ the Bi-LSTM to learn the reliable dependency between spatio-temporal features
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=1)

    def forward(self, x):
        b, c, T = x.size() # (30, 32, 124)
        x = x.view(x.size(-1), -1, c)  # (b, c, T) -> (T, b, c) (124, 30, 32)
        r_out, _ = self.rnn(x)  # r_out shape [time_step, batch_size, output_size*2] (124, 30, 64)
        out = r_out.view(b, 2 * T * c, -1) # 改变向量形状 (30, 7936, 1)
        return out


class ESNet(nn.Module):
    def calculateOutSize(self, model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is a array
        '''
        data = torch.randn(1, 1, nChan, nTime)
        out = model(data).shape
        return out[1:]

    # 空间滤波器模块，为每个通道分配不同的权重并融合它们
    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
        '''
        block = []
        block.append(Constraint.Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                                     max_norm=1.0)) # (30, 16, 1, 256)
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.PReLU())
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
                               stride=(1, stride))) # input(30, 16, 8, 256) output(30, 32, 8, 124)
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def __init__(self, num_channels, T, num_classes):
        super(ESNet, self).__init__()
        self.dropout_level = 0.5
        self.F = [num_channels * 2] + [num_channels * 4]
        self.K = 10
        self.S = 2

        net = []
        net.append(self.spatial_block(num_channels, self.dropout_level)) # （30， 16， 1， 256）
        net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level,
                                           self.K, self.S)) # (30, 32, 1, 124)

        self.conv_layers = nn.Sequential(*net)

        self.fcSize = self.calculateOutSize(self.conv_layers, num_channels, T) # (32, 1, 124)
        self.fcUnit = self.fcSize[0] * self.fcSize[1] * self.fcSize[2] * 2 # 7936
        self.D1 = self.fcUnit // 10 # 793
        self.D2 = self.D1 // 5 # 158

        self.rnn = LSTM(input_size=self.F[1], hidden_size=self.F[1])

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fcUnit, self.D1),
            nn.PReLU(),
            nn.Linear(self.D1, self.D2),
            nn.PReLU(),
            nn.Dropout(self.dropout_level),
            nn.Linear(self.D2, num_classes))

    def forward(self, x):
        # print(x.shape)
        # input: (30, 1, 8, 256) 30个样本，1，8个通道，256个时间点
        out = self.conv_layers(x) # (30, 32, 1, 124)
        out = out.squeeze(2) # (30, 32, 124)
        # print(out.shape)
        r_out = self.rnn(out) # (30, 7936, 1)
        out = self.dense_layers(r_out) # (30, 12)
        return out
