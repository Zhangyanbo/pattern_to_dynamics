from train_diffusion import Diffusion
from distributions import TwoPeaksDataset, RingDataset, LorenzDataset, TwoMoonsDataset
import torch
from utils import VPJBatchNorm, integrate_rk4, div_estimate
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def load_model(name):
    if name == 'two_peaks':
        dataset = TwoPeaksDataset()
    elif name == 'ring':
        dataset = RingDataset()
    elif name == 'lorenz':
        dataset = LorenzDataset()
    elif name == 'two_moons':
        dataset = TwoMoonsDataset()
    else:
        raise ValueError(f"Model {name} not supported")

    score_model = Diffusion(dim=dataset.dim)
    if name == 'two_peaks':
        score_model.load_state_dict(torch.load('./models/two_peaks_ddim.pth'))
    elif name == 'ring':
        score_model.load_state_dict(torch.load('./models/ring_ddim.pth'))
    elif name == 'lorenz':
        score_model.load_state_dict(torch.load('./models/lorenz_ddim.pth'))
    elif name == 'two_moons':
        score_model.load_state_dict(torch.load('./models/two_moons_ddim.pth'))

    return score_model, dataset


def plot_flow(flow, score_model, dataloader, device):
    # plot the flow
    x = torch.linspace(-3, 3, 20)
    y = torch.linspace(-3, 3, 20)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    flow.eval()

    with torch.no_grad():
        v = flow.forward(points).cpu() * 0.025

    plt.figure(figsize=(5, 5))

    samples = next(iter(dataloader))
    samples = samples.cpu()
    plt.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.5)
    plt.quiver(X.numpy(), Y.numpy(), v[:, 0].numpy(), v[:, 1].numpy(), 
            angles='xy', scale_units='xy', scale=0.2, alpha=0.5)


    x0 = samples[0].unsqueeze(0).to(device)
    trajectory = []
    for i in range(500):
        with torch.no_grad():
            x0 = integrate_rk4(lambda x: flow.forward(x) + score_model.score(x, t=0.1) * 0.0, x0, dt=0.1)
            trajectory.append(x0)
    trajectory = torch.cat(trajectory, dim=0).cpu()
    plt.plot(trajectory[:, 0], trajectory[:, 1], '-', alpha=1)


    plt.title("Flow")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)


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

def train_dynamics(score_model, dataset, batch_size=2048, model='two_peaks', num_samples=10, lr=1e-3, weight_decay=1e-5, num_epochs=1024, device='cpu', dim_hidden=64, noise=0.1, num_kernels=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    score_model.to(device)

    flow = FlowKernel(dim=dataset.dim, dim_hidden=dim_hidden, num_kernels=num_kernels)
    # flow = Flow(dim=dataset.dim, dim_hidden=dim_hidden)
    flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    div_losses = []
    oth_losses = []

    for epoch in tqdm(range(num_epochs)):
        acc_loss = 0
        acc_div_loss = 0
        acc_oth_loss = 0
        for x in dataloader:
            x = x.to(device)
            x = x + torch.randn_like(x) * noise
            optimizer.zero_grad()
            div, s, v, div_term, oth_term = div_estimate(flow, score_model, x, num_samples=num_samples)
            loss = torch.mean(div ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            optimizer.step()
            acc_loss += loss.item()

            acc_div_loss += (div_term ** 2).mean().item()
            acc_oth_loss += (oth_term ** 2).mean().item()
        losses.append(acc_loss / len(dataloader))
        div_losses.append(acc_div_loss / len(dataloader))
        oth_losses.append(acc_oth_loss / len(dataloader))

    return flow, losses, div_losses, oth_losses

def make_plot_folder(model):
    if not os.path.exists(f'./figure/{model}'):
        os.makedirs(f'./figure/{model}')
    root_path = f'./figure/{model}'
    return root_path

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='two_peaks')
    parser.add_argument('--num_epochs', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dim_hidden', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--num_kernels', '-k', type=int, default=4)
    args = parser.parse_args()

    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    score_model, dataset = load_model(args.model)
    device = torch.device(args.device)
    flow, losses, div_losses, oth_losses = train_dynamics(score_model, dataset, **vars(args))
    torch.save(flow.state_dict(), f'./models/{args.model}_flow.pth')

    root_path = make_plot_folder(args.model)

    plt.plot(losses, label='total loss')
    plt.plot(div_losses, label='div loss')
    plt.plot(oth_losses, label='oth loss')
    plt.semilogy()
    plt.legend()
    plt.savefig(os.path.join(root_path, 'loss.png'))
    plt.savefig(os.path.join(root_path, 'loss.pdf'))
    plt.close()

    plot_flow(flow, score_model, DataLoader(dataset, batch_size=256, shuffle=True), device)
    plt.savefig(os.path.join(root_path, 'flow.png'))
    plt.savefig(os.path.join(root_path, 'flow.pdf'))
    plt.close()