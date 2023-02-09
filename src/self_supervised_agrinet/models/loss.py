import torch
import torch.nn as nn

class BarlowTwinsLoss(nn.Module):

    def __init__(self, lambdap):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambdap

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # normalize repr. along the batch dimension
        z_a = z[0:len(z):2] # this way z_a and z_b have at same index the aug of same image
        z_b = z[1:len(z):2]
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        try:
            D = z_a.size(1)
        except: 
            D = N 

        #import ipdb; ipdb.set_trace()
        # cross-correlation matrix
        c = z_a_norm.T @ z_b_norm / N # DxD
        # loss
        c_diff = (c - torch.eye(D).cuda()).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
    
        return loss
