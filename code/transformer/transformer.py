import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, vocab_size, feature_size):
        super(Transformer, self).__init__()

        # self.embedding = nn.Embedding(1452, 10)
        self.layers = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1)
        self.transformer = nn.TransformerEncoder(self.layers, num_layers=6)
        self.decoder = nn.Linear(feature_size, vocab_size)

    def forward(self, x):
        # X shape: [seq_len, batch_size]
        print(x.shape)

        # x = self.embedding(x)
        # X shape: [seq_len, batch_size, feature_size]
        print(x.shape)
        x = self.transformer(x)
        # X shape: [seq_len, batch_size, feature_size]
        print(x.shape)
        x = self.decoder(x)
        # X shape: [seq_len, batch_size, vocab_size]
        print(x.shape)

        return x


# src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).view(10, 1)
