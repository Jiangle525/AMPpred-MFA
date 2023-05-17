from torch import nn
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
        self.conv_out_channels = 4                      # 卷积层输出通道数
        self.lstm_input_size = self.embedding_dim       # LSTM输入大小
        self.lstm_hidden_dim = 32                       # LSTM隐藏层特征数目
        self.lstm_num_layers = 2                        # LSTM循环次数
        self.lstm_dropout = 0.5                         # LSTM丢弃率
        self.lstm_bidirectional = True                  # LSTM是否为双向
        self.dropout = 0.5                              # 注意力丢弃率
        self.num_heads = 4                              # 多头注意力的多头数


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.embedding = None
        if config.feature_method in ['vocab', 'mixed']:
            self.embedding = nn.Embedding(
                config.vocab_size, config.embedding_dim, padding_idx=config.embed_padding_idx)
            config.conv_out_channels = config.padding_size
        else:
            self.conv = nn.Conv1d(1, config.conv_out_channels, 1)
            config.lstm_input_size = config.feature_dim

        self.position_encoding = MultiAttention.Positional_Encoding(
            config.lstm_input_size, config.conv_out_channels, config.dropout, config.device)
        self.attention = MultiAttention.Model(
            config.lstm_input_size, config.num_heads)
        self.lstm = nn.LSTM(config.lstm_input_size, config.lstm_hidden_dim, config.lstm_num_layers,
                            bidirectional=config.lstm_bidirectional, batch_first=True, dropout=config.lstm_dropout)
        if config.num_classes:
            self.fc = nn.Linear(config.lstm_hidden_dim * config.lstm_num_layers *
                                config.conv_out_channels, config.num_classes)
            # self.fc = nn.Linear(config.lstm_hidden_dim * config.lstm_num_layers, config.num_classes)
        else:
            self.fc = None

    def forward(self, x):
        x = x[0]
        if self.embedding:
            out = self.embedding(x)
        else:
            x = x.unsqueeze(1)
            out = self.conv(x)
        self.attention_inputs = out
        out = self.position_encoding(out)
        out, self.attention_wight = self.attention(out)
        self.attention_outputs = out
        # out.shape: (batch_size, config.conv_in_channels, config.feature_dim)
        # attention.shape: (batch_size * config.num_heads, config.conv_in_channels, config.conv_in_channels)
        # 用vocab编码:   out.shape: [64, 100, 60] attention.shape: [256, 100, 100]
        out, _ = self.lstm(out)
        out = out.reshape(out.size(0), -1)
        if self.fc:
            out = self.fc(out)
        return out
