import torch
from torch import nn
from .Model import BaseConfig
from . import Att_BiLSTM, Att_ConvNet


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        # Model hyperparameters
        self.learning_rate = 6e-5                       # Learning rate
        self.batch_size = 64                            # Batch size
        self.num_epochs = 300                           # Training epoch
        self.num_patience = 20                          # Early Stopping Tolerance Times

        self.config_manual_feature = Att_ConvNet.Config()
        self.config_manual_feature.num_classes = 1200

        self.config_vocab_feature = Att_BiLSTM.Config()
        self.config_vocab_feature.num_classes = 1200


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        
        config.config_manual_feature.feature_dim = config.feature_dim

        config.config_vocab_feature.feature_method = 'vocab'
        config.config_vocab_feature.vocab_size = config.vocab_size
        config.config_vocab_feature.embed_padding_idx = config.embed_padding_idx

        self.net1 = Att_ConvNet.Model(config.config_manual_feature)
        self.net2 = Att_BiLSTM.Model(config.config_vocab_feature)
        self.fc = nn.Linear(config.config_vocab_feature.num_classes +
                            config.config_manual_feature.num_classes, config.num_classes)

    def forward(self, x):
        x1, x2 = x[0], x[1]
        out1 = self.net1([x1])
        out2 = self.net2([x2])
        self.attention_wight1 = self.net1.attention_wight
        self.attention_wight1_inputs = self.net1.attention_inputs
        self.attention_wight1_outputs = self.net1.attention_outputs
        self.attention_wight2 = self.net2.attention_wight
        self.attention_wight2_inputs = self.net2.attention_inputs
        self.attention_wight2_outputs = self.net2.attention_outputs
        out = torch.cat([out1, out2], dim=1)
        self.last_feature = out
        out = self.fc(out)
        return out
