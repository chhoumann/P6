from data.get_data import DataType, get_data
from transformer.transformer import Transformer
from transformer.Time2Vector import Time2Vector
import tensorflow as tf

import torch

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

if __name__ == "__main__":
    print("Starting")
    data = get_data(DataType.delhi_small)

    # src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).view(10, 1)
    # print(data["X_train"][0].view(10, 1))
    # print(data["X_train"])


    time_embedding = Time2Vector(10)
    x = data["X_train"]
    # x = torch.tensor(x).to(torch.int64)
    # x = tf.cast(x, tf.float32)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = time_embedding(x)
    x = x.numpy()
    x = torch.tensor(x)

    print(x)
    transformer = Transformer(1452, 2)
    pred = transformer(x)
    # print(pred)
    criterion = RMSELoss
    loss = criterion(pred, data["y_train"])
    print(loss)

