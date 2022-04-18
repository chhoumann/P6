from typing import TypedDict
import torch


class DataDict(TypedDict):
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
