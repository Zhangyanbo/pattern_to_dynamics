import torch
import torch.nn as nn
from utils import VPJBatchNorm


class Flow(nn.Module):
    def __init__(self, dim=2, dim_hidden=64):
        super(Flow, self).__init__()
        self.nonlinear = nn.SiLU()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            VPJBatchNorm(dim_hidden),
            self.nonlinear,
            nn.Linear(dim_hidden, dim_hidden),
            VPJBatchNorm(dim_hidden),
            self.nonlinear,
            nn.Linear(dim_hidden, dim_hidden),
            VPJBatchNorm(dim_hidden),
            self.nonlinear,
            nn.Linear(dim_hidden, dim),
            VPJBatchNorm(dim, affine=False)
        )
    
    def forward(self, x):
        output = self.mlp(x)
        return output

class FlowKernel(nn.Module):
    def __init__(self, dim=2, dim_hidden=64, num_kernels=4):
        super(FlowKernel, self).__init__()
        self.num_kernels = num_kernels
        self.nonlinear = nn.SiLU()
        self.mlp = nn.Sequential(
            nn.Linear(dim * (2 * num_kernels + 1), dim_hidden),
            VPJBatchNorm(dim_hidden),
            self.nonlinear,
            nn.Linear(dim_hidden, dim_hidden),
            VPJBatchNorm(dim_hidden),
            self.nonlinear,
            nn.Linear(dim_hidden, dim),
            VPJBatchNorm(dim, affine=False)
        )
    
    def position_encoding(self, x):
        """
        Apply positional encoding to input coordinates as in NeRF.
        For each dimension, adds sin/cos encodings at different frequencies
        while preserving the original coordinate.
        
        Args:
            x: Input tensor of shape [batch, dim]
            
        Returns:
            Encoded tensor with shape [batch, dim * (2 * num_kernels + 1)]
        """
        batch_size, dim = x.shape
        
        # Initialize output tensor that will include original coordinates
        encoded = [x]
        
        # Apply encoding for each frequency
        for i in range(self.num_kernels):
            # 2^i gives increasing frequency for each level
            freq = 2.0 ** i
            
            # Add sin and cos encodings for each dimension
            sin_encoding = torch.sin(x * freq)
            cos_encoding = torch.cos(x * freq)
            
            encoded.append(sin_encoding)
            encoded.append(cos_encoding)
        
        # Concatenate all encodings along the feature dimension
        return torch.cat(encoded, dim=-1)
    
    def forward(self, x):
        x = self.position_encoding(x)
        output = self.mlp(x)
        return output

class FlowAugmented(nn.Module):
    def __init__(self, score_model, dim=2, dim_hidden=64):
        super(FlowAugmented, self).__init__()
        self.score_model = score_model
        self.nonlinear = nn.SiLU()
        self.mlp = nn.Sequential(
            nn.Linear(dim + score_model.dim, dim_hidden),
            VPJBatchNorm(dim_hidden),
            self.nonlinear,
            nn.Linear(dim_hidden, dim_hidden),
            VPJBatchNorm(dim_hidden),
            self.nonlinear,
            nn.Linear(dim_hidden, dim),
            VPJBatchNorm(dim, affine=False)
        )
    
    def parameters(self):
        # not include score_model parameters
        return self.mlp.parameters()
    
    def forward(self, x):
        score = self.score_model.score(x, t=0.1)
        input = torch.cat([x, score], dim=-1)
        return self.mlp(input)