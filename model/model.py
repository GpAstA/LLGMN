import torch.nn as nn
from .LLGMN import LLGMN

class ThroughNet(nn.Module):
    def __init__(self, input_dim):
        super(ThroughNet, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        nn.init.ones_(self.fc.weight)
        
    def forward(self, x):
        return self.fc(x)

def build_model(input_dim, n_class, n_component):
    return nn.Sequential(
        ThroughNet(input_dim),
        LLGMN(in_futures=input_dim, n_class=n_class, n_component=n_component)
    )
