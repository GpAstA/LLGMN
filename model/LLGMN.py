import torch
import torch.nn as nn

class LLGMN(nn.Module):
    def __init__(self, in_features, n_class, n_component=1):
        super(LLGMN, self).__init__()
        self.in_features = in_features
        self.n_class = n_class
        self.n_component = n_component
        self.H = int(1 + self.in_features * (self.in_features + 3) / 2)
        self.weight = nn.Parameter(torch.Tensor(self.H, self.n_class, self.n_component))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        with torch.no_grad():
            self.weight[:, self.n_class - 1, self.n_component - 1] = 0

    def nonlinear_trans(self, x):
        if self.in_features == 1:
            quadratic_term = x * x
        else:
            outer_prod = torch.einsum('ni,nj->nij', x, x)
            mask = torch.triu(torch.ones(self.in_features, self.in_features)).bool()
            quadratic_term = outer_prod[:, mask]
        bias_term = torch.ones_like(torch.sum(x, dim=1, keepdim=True))
        output = torch.cat([bias_term, x, quadratic_term], dim=1)
        return output

    def redundant_term_to_zero(self, I2):
        I2_reshaped = I2.view(-1, self.n_class * self.n_component)
        I2_reshaped[:, -1] = 0.0
        output = I2_reshaped.view(-1, self.n_class, self.n_component)
        return output

    def forward(self, inputs):
        x_nonlinear = self.nonlinear_trans(inputs)
        I2 = torch.einsum('ni,icm->ncm', x_nonlinear, self.weight)
        I2_ = self.redundant_term_to_zero(I2)
        exp_I2 = torch.exp(I2_)
        denominator = torch.sum(exp_I2, dim=[1, 2], keepdim=True)
        denominator = denominator.expand(-1, self.n_class, self.n_component)
        O2 = exp_I2 / denominator
        O3 = torch.sum(O2, dim=2, keepdim=False)

        return O3
