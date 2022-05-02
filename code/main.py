from data.get_data import DataType, get_data
from transformer.transformer import Transformer

import torch

if __name__ == "__main__":
    print("Starting")
    data = get_data(DataType.delhi_small)

    transformer = Transformer(1452, 10)
    # src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).view(10, 1)
    # print(data["X_train"][0].view(10, 1))
    print(data["X_train"])
    x = torch.tensor(data["X_train"]).to(torch.int64)
    print(x)
    # transformer(x)
    # print(pred)

