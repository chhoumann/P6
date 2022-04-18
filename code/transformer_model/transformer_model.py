import math

import torch
import torch.nn as nn
from matplotlib import pyplot

import model_data
import settings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerAttentionModel(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransformerAttentionModel, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def predict_future(self, data_source, steps):
        self.eval()
        total_loss = 0.
        test_result = torch.Tensor(0)

        truth = torch.Tensor(0)
        _, data = model_data.get_batch(data_source, 0, 1)

        with torch.no_grad():
            for i in range(0, steps, 1):
                input = torch.clone(data[-settings.input_window:])
                input[-settings.output_window:] = 0
                output = self(data[-settings.input_window:])  # TODO: Is this possible?
                data = torch.cat((data, output[-1:]))

        data = data.cpu().view(-1)

        pyplot.plot(data, color="red")
        pyplot.plot(data[:settings.input_window], color="blue")
        pyplot.grid(True, which='both')
        pyplot.axhline(y=0, color='k')
        pyplot.savefig('./graphs/transformer-future%d.png' % steps)
        pyplot.close()

    #  either there is an error in the loss or in the train method, but the results are different from those of predict_future
    def evaluate(self, data_source):
        self.eval()  # Turn on the evaluation mode
        total_loss = 0.
        eval_batch_size = 1000

        with torch.no_grad():
            for i in range(0, len(data_source) - 1, eval_batch_size):
                data, targets = model_data.get_batch(data_source, i, eval_batch_size)
                output = self(data)

                if settings.calculate_loss_over_all_values:
                    total_loss += len(data[0]) * settings.criterion(output, targets).cpu().item()
                else:
                    total_loss += len(data[0]) * settings.criterion(output[-settings.output_window:],
                                                                    targets[-settings.output_window:]).cpu().item()
        return total_loss / len(data_source)
