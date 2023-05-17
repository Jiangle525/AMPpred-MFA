from torch import nn
import torch.nn.functional as F
from .Model import BaseConfig
from . import MultiAttention


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        # 训练超参数
        self.learning_rate = 6e-5                       # 学习率
        self.batch_size = 64                            # 批量大小
        self.num_epochs = 300                           # 训练次数
        self.num_patience = 20                          # 早停法忍耐次数

        # 模型超参数
        self.conv_in_channels = 1                       # 卷积层输入通道数
        self.conv_out_channels = 4                      # 卷积层输出通道数
        self.dropout = 0.5                              # 丢弃率
        self.num_heads = 4


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.embedding = None
        if config.feature_method in ['vocab']:
            self.embedding = nn.Embedding(
                config.vocab_size, config.embedding_dim, padding_idx=config.embed_padding_idx)
            config.conv_in_channels = config.padding_size
            config.feature_dim = config.embedding_dim

        self.position_encoding = MultiAttention.Positional_Encoding(
            config.feature_dim, config.conv_in_channels, config.dropout, config.device)
        self.attention = MultiAttention.Model(
            config.feature_dim, config.num_heads, config.dropout)
        self.conv1 = nn.Conv1d(in_channels=config.conv_in_channels,
                               out_channels=config.conv_out_channels*2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=config.conv_out_channels*2)
        self.conv2 = nn.Conv1d(in_channels=config.conv_out_channels*2,
                               out_channels=config.conv_out_channels*4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=config.conv_out_channels*4)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(config.dropout)
        if config.num_classes:
            self.fc = nn.Linear(config.conv_out_channels *
                                config.feature_dim, config.num_classes)
        else:
            self.fc = None

    def forward(self, x):
        x = x[0]
        if self.embedding:
            out = self.embedding(x)
        else:
            out = x.unsqueeze(1)
        self.attention_inputs = out
        out = self.position_encoding(out)
        out, self.attention_wight = self.attention(out)
        self.attention_outputs = out
        # out.shape: (batch_size, config.conv_in_channels, config.feature_dim)
        # attention.shape: (batch_size * config.num_heads, config.conv_in_channels, config.conv_in_channels)
        # 用vocab编码:   out.shape: [64, 100, 60] attention.shape: [256, 100, 100]
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = out.reshape(out.size(0), -1)
        if self.fc:
            out = self.fc(out)
        return out
