import torch.nn as nn
import torch

# This concept is also called teacher forcing.
# The flag decides if the loss will be calculated over all or just the predicted values.
calculate_loss_over_all_values = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)

input_window = 100
output_window = 5
batch_size = 10
epochs = 100  # The number of epochs
learning_rate = 0.005

criterion = nn.MSELoss()
