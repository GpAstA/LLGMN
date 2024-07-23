# src/through_net.py

import torch
import torch.nn as nn
from .elasticnet_reducer import ElasticNetDimensionReducer

class ImprovedThroughNet(nn.Module):
    def __init__(self, input_shape, output_dim, alpha=1.0, l1_ratio=0.5):
        super(ImprovedThroughNet, self).__init__()
        self.reducer = ElasticNetDimensionReducer(input_shape[-1], output_dim, alpha, l1_ratio)

    def forward(self, x):
        return self.reducer(x)

    def get_elasticnet_penalty(self):
        return self.reducer.elasticnet_penalty()
