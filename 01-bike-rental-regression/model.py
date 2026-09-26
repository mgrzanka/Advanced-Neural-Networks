from typing import Any

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        out = self.input_layer(X)
        out = self.relu(out)
        out = self.output_layer(out)
        return out
