from torch import nn


class SELayer_Linear(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid, norm=False):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()
        self.norm = norm

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class MLN(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta
        return out


class MLN2(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.LayerNorm(f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Sequential(
            nn.Linear(f_dim, f_dim),
            nn.Sigmoid(),
        )
        self.beta = nn.Sequential(
            nn.Linear(f_dim, f_dim),
            nn.LayerNorm(f_dim),
        )
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma[0].weight)
        nn.init.zeros_(self.beta[0].weight)
        nn.init.ones_(self.gamma[0].bias)
        nn.init.zeros_(self.beta[0].bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta
        return out