import torch
import torch.nn as nn


class SCU(nn.Module):

    def __init__(self, opt, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(

            # nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
            nn.Conv1d(in_channels=9, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm1d(num_features=32),  # 对于一维批量归一化，通常等于卷积层的输出通道数
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(opt.dropout_level))

        self.dense_layers = nn.Sequential(
            # nn.Linear(5984, 600),
            # nn.Linear(7984, 600),
            # nn.Linear(4240, 600),
            # nn.Linear(11984, 1200),
            nn.Linear(11968, 1200),
            nn.ReLU(),
            nn.Dropout(opt.dropout_level),
            nn.Linear(1200, 120),
            nn.ReLU(),
            nn.Dropout(opt.dropout_level),
            nn.Linear(120, num_classes))

    def forward(self, x):
        # print("scu input",x.shape)
        out = self.layer1(x)
        # print("scu layer1 output", x.shape)
        out = out.view(out.size(0), -1)
        # print("scu out.view output", out.shape)
        out = self.dense_layers(out)
        # print("scu output",out)
        return out
