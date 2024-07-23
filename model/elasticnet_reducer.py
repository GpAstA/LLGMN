# src/elasticnet_reducer.py

import torch
import torch.nn as nn

class ElasticNetDimensionReducer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=1.0, l1_ratio=0.5):
        super(ElasticNetDimensionReducer, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def forward(self, inputs):
        return self.dense(inputs)

    def elasticnet_penalty(self):
        l1_penalty = torch.sum(torch.abs(self.dense.weight))
        l2_penalty = torch.sum(torch.square(self.dense.weight))
        return self.alpha * (self.l1_ratio * l1_penalty + (1 - self.l1_ratio) * l2_penalty)
