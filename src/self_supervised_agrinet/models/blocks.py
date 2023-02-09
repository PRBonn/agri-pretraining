import torch.nn as nn


class MyBlock(nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LeakyReLU())

    def forward(self,x):
        return self.net(x)