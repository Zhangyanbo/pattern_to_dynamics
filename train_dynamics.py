from train_diffusion import Diffusion
from distributions import TwoPeaksDataset, RingDataset, LorenzDataset
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
    
    score_model = Diffusion(dim=dataset.dim)
    if name == 'two_peaks':
        score_model.load_state_dict(torch.load('./models/two_peaks_ddim.pth'))
    elif name == 'ring':
        score_model.load_state_dict(torch.load('./models/ring_ddim.pth'))
    elif name == 'lorenz':
        score_model.load_state_dict(torch.load('./models/lorenz_ddim.pth'))
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


    x0 = torch.Tensor([[1.5, 1.5]]).to(device)
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
    def __init__(self, dim=2):
        super(Flow, self).__init__()
        self.nonlinear = nn.SiLU()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64),
            VPJBatchNorm(64),
            self.nonlinear,
            nn.Linear(64, 64),
            VPJBatchNorm(64),
            self.nonlinear,
            nn.Linear(64, dim),
            VPJBatchNorm(dim, affine=False)
        )
    
    def forward(self, x):
        output = self.mlp(x)
        return output

def train_dynamics(score_model, dataset, batch_size=2048, num_samples=10, lr=1e-3, weight_decay=1e-5, num_epochs=1024, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    score_model.to(device)

    flow = Flow(dim=dataset.dim)
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
            x = x + torch.randn_like(x) * 0.1
            optimizer.zero_grad()
            div, s, v, div_term, oth_term = div_estimate(flow, score_model, x, num_samples=10)
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


if __name__ == '__main__':
    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    score_model, dataset = load_model('two_peaks')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow, losses, div_losses, oth_losses = train_dynamics(score_model, dataset, num_epochs=1024, device=device)
    torch.save(flow.state_dict(), './models/two_peaks_flow.pth')

    plt.plot(losses, label='total loss')
    plt.plot(div_losses, label='div loss')
    plt.plot(oth_losses, label='oth loss')
    plt.semilogy()
    plt.legend()
    plt.savefig('./figure/two_peaks_dynamics_loss.png')
    plt.savefig('./figure/two_peaks_dynamics_loss.pdf')
    plt.close()

    plot_flow(flow, score_model, DataLoader(dataset, batch_size=256, shuffle=True), device)
    plt.savefig('./figure/two_peaks_dynamics_flow.png')
    plt.savefig('./figure/two_peaks_dynamics_flow.pdf')
    plt.close()