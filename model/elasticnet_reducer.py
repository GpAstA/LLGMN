import torch
import torch.nn as nn

class ElasticNetDimensionReducer(nn.Module):
    def __init__(self, input_dim, alpha=1.0, l1_ratio=0.5):
        super(ElasticNetDimensionReducer, self).__init__()
        self.input_dim = input_dim
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.dense = nn.Linear(input_dim, input_dim)  

    def forward(self, inputs):
        return self.dense(inputs)

    def elasticnet_penalty(self):
        l1_penalty = torch.sum(torch.abs(self.dense.weight))
        l2_penalty = torch.sum(torch.square(self.dense.weight))
        return self.alpha * (self.l1_ratio * l1_penalty + (1 - self.l1_ratio) * l2_penalty)

    def reduce_dimension(self, threshold=1e-3):
        with torch.no_grad():
            weight_magnitudes = torch.abs(self.dense.weight).sum(dim=1)
            important_indices = weight_magnitudes > threshold
            reduced_weight = self.dense.weight[:, important_indices]
            self.dense = nn.Linear(self.input_dim, important_indices.sum().item(), bias=False)
            self.dense.weight = nn.Parameter(reduced_weight)
