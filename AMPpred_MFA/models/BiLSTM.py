from torch import nn
from .Model import BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        # Model hyperparameters
        self.learning_rate = 6e-5                       # Learning rate
        self.batch_size = 64                            # Batch size
        self.num_epochs = 300                           # Training epochs
        self.num_patience = 20                          # Early Stopping Tolerance Times

        self.conv_out_channels = 4                      # The number of convolutional layer output channels
        self.lstm_input_size = self.embedding_dim       # LSTM input size
        self.lstm_hidden_dim = 32                       # Number of LSTM hidden layer features
        self.lstm_num_layers = 2                        # LSTM cycle times
        self.lstm_dropout = 0.5                         # LSTM dropout rate
        self.lstm_bidirectional = True                  # Whether LSTM is bidirectional


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
        out, _ = self.lstm(out)
        out = out.reshape(out.size(0), -1)
        if self.fc:
            out = self.fc(out)
        return out
