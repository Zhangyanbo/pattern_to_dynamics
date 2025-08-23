import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

from probflow import Diffuser, ResNet2D, VPJBatchNorm2d, VPJBatchNorm, div_estimate
from dynamics import Trainer
from turing_pattern import GrayScottSimulator, create_random_state, TuringPatternDataset
from diffusers import UNet2DModel, DDPMScheduler
from circular import UNet2DModelWithPadding


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_score_model(name:str, device:str='cuda', freeze:bool=True) -> nn.Module:
    model = UNet2DModelWithPadding.from_pretrained(f"./turing_pattern/diffusion_models/{name}")
    scheduler = DDPMScheduler.from_pretrained(f"./turing_pattern/diffusion_models/{name}")
    model.eval()
    model.to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model, scheduler


class SymConv2d_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SymConv2d_3, self).__init__()
        self.register_buffer('mask_center', torch.Tensor(
            [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]))
        
        self.register_buffer('mask_corner', torch.Tensor(
            [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]]))
        self.register_buffer('mask_cross', torch.Tensor(
            [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]]))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.padding = 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.center = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.corner = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.cross = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.init_weights()

    @torch.no_grad()
    def init_weights(self, nonlinearity='relu', a=0.0):
        gain = nn.init.calculate_gain(nonlinearity, a)
        fan_in = self.in_channels * 9
        bound = (gain**2 / (self.in_channels * 9) * 3) ** 0.5

        nn.init.uniform_(self.center, -bound, bound)
        nn.init.uniform_(self.corner, -bound, bound)
        nn.init.uniform_(self.cross,  -bound, bound)
        nn.init.zeros_(self.bias)
    
    @property
    def kernel(self):
        k = torch.einsum('oi, hw -> oihw', self.center, self.mask_center) + \
        torch.einsum('oi, hw -> oihw', self.corner, self.mask_corner) + \
        torch.einsum('oi, hw -> oihw', self.cross, self.mask_cross)
        return k
    
    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        # Circularly pad the input on all four sides, then run a valid (no‑pad) conv
        x = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
            mode="circular",
        )
        return F.conv2d(x, self.kernel, bias=self.bias, padding=0, groups=1)

    def forward(self, x):
        x = self._conv(x)
        return x


class FlowModel(nn.Module):
    def __init__(self, hidden_channels=32):
        super(FlowModel, self).__init__()
        self.cnn = nn.Sequential(
            SymConv2d_3(2, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1, padding=0),
        )
        self.bn = VPJBatchNorm2d(2, affine=False)

        # self.skip_connection = nn.Conv2d(2, 2, kernel_size=1, padding=0)
    
    def forward(self, x):
        # return self.bn(self.cnn(x))
        return self.cnn(x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a flow model for Turing patterns.')
    parser.add_argument('--dataset', type=str, default='spirals',
                        help='Dataset to use for training the flow model.')
    parser.add_argument('-t', '--dt', type=float, default=10,
                        help='Alpha value for the diffusion model.')
    parser.add_argument('-s', '--schedule', action='store_true',
                        help='Use cosine schedule for learning rate.')
    parser.add_argument('-g', '--gaussian_weight', type=float, default=0.0,
                        help='Weight for the Gaussian term in the score model.')
    parser.add_argument('--sym', type=float, default=0.0,
                        help='Symmetry penalty weight for the loss function.')
    parser.add_argument('--epochs', '-e', type=int, default=200,
                        help='Number of epochs to train the flow model.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for initialization.')

    args = parser.parse_args()

    setup_seed(args.seed)

    dataset = TuringPatternDataset.load(f'./turing_pattern/data/{args.dataset}_128x128.pt')
    score_model, scheduler = load_score_model(args.dataset, device='cuda')
    flow_model = FlowModel().to('cuda')

    trainer = Trainer(
            flow_model, 
            score_model, 
            scheduler,
            (2, 128, 128), # It is very important to set the correct input shape
            dataset=dataset, 
            model=args.dataset,
            use_wandb=True, 
            lr=1e-2, 
            num_samples=8, 
            weight_decay=1e-5, 
            gradient_accumulation_steps=8,
            schedule=args.schedule,
            gaussian_weight=args.gaussian_weight,
            symmetry_punalty=args.sym,
            diffuse_time_t=args.dt,
        )

    trainer.train(epochs=args.epochs, batch_size=128)
    # save flow_model
    import os
    os.makedirs('./turing_pattern/flow_models/', exist_ok=True)
    # save the model state dict
    torch.save(flow_model.state_dict(), f'./turing_pattern/flow_models/{args.dataset}.pth')