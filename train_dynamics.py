from train_diffusion import Diffusion
from distributions import TwoPeaksDataset, RingDataset, LorenzDataset, TwoMoonsDataset
import torch
from utils import VPJBatchNorm, integrate_rk4, div_estimate
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from models import FlowKernel, Flow

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
    path = f'./results/{name}/diffusion_model.pth'
    score_model.load_state_dict(torch.load(path))

    return score_model, dataset


def plot_flow2d(flow, score_model, dataloader, device):
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

def plot_flow3d(flow, score_model, dataloader, device):
    """
    Plot the flow in 3D space.
    
    Args:
        flow: The flow model to visualize
        score_model: The score model
        dataloader: DataLoader containing the dataset
        device: Device to run computations on
    """
    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get data samples
    samples = next(iter(dataloader))
    samples = samples.cpu()
    
    # Plot data points
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.5, s=10)
    
    # Create grid for vector field
    grid_size = 8
    limit = 3
    x = torch.linspace(-limit, limit, grid_size)
    y = torch.linspace(-limit, limit, grid_size)
    z = torch.linspace(-limit, limit, grid_size)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).to(device)
    
    # Compute vector field
    flow.eval()
    with torch.no_grad():
        vectors = flow.forward(grid_points).cpu() * 0.025
    
    # Plot vector field (every few points to avoid overcrowding)
    skip = 4
    points = grid_points.cpu().numpy()
    vecs = vectors.cpu().numpy()
    ax.quiver(
        points[::skip, 0], points[::skip, 1], points[::skip, 2],
        vecs[::skip, 0], vecs[::skip, 1], vecs[::skip, 2],
        length=0.5, normalize=True, alpha=0.25, color='black'
    )
    
    # Plot trajectory
    x0 = samples[0].unsqueeze(0).to(device)
    trajectory = []
    for i in range(500):
        with torch.no_grad():
            x0 = integrate_rk4(
                lambda x: flow.forward(x) + score_model.score(x, t=0.1) * 0.0, 
                x0, 
                dt=0.1
            )
            trajectory.append(x0)
    
    trajectory = torch.cat(trajectory, dim=0).cpu().numpy()
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], '-', alpha=1, linewidth=2, color='orange')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Flow Visualization')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)


def plot_flow(flow, score_model, dataloader, device):
    if flow.dim == 2:
        plot_flow2d(flow, score_model, dataloader, device)
    elif flow.dim == 3:
        plot_flow3d(flow, score_model, dataloader, device)
    else:
        raise ValueError(f"Flow dimension {flow.dim} not supported")

def energy_loss(v):
    energy = torch.sum(v ** 2, dim=-1)
    return torch.exp(-energy).mean()

def train_dynamics(score_model, dataset, batch_size=2048, model='two_peaks', num_samples=10, lr=1e-3, weight_decay=1e-5, num_epochs=1024, device='cpu', dim_hidden=64, noise=0.1, num_kernels=4, non_kernel=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    score_model.to(device)

    if non_kernel:
        flow = Flow(dim=dataset.dim, dim_hidden=dim_hidden)
    else:
        flow = FlowKernel(dim=dataset.dim, dim_hidden=dim_hidden, num_kernels=num_kernels)
    flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    div_losses = []
    oth_losses = []
    energy_losses = []
    t = 0.2
    print(f"Alpha.sqrt(): {score_model.alpha_t(torch.tensor(t, device=device)).sqrt().item()}")

    for epoch in tqdm(range(num_epochs)):
        acc_loss = 0
        acc_div_loss = 0
        acc_oth_loss = 0
        acc_energy_loss = 0
        for x in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            div, s, v, div_term, oth_term = div_estimate(flow, score_model, x, t, num_samples=num_samples)
            loss_energy = torch.mean(energy_loss(v))
            loss = torch.mean(div ** 2) + energy_loss_weight * loss_energy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            optimizer.step()
            acc_loss += loss.item()

            acc_div_loss += div_term.mean().item()
            acc_oth_loss += oth_term.mean().item()
            acc_energy_loss += loss_energy.item()
        losses.append(acc_loss / len(dataloader))
        div_losses.append(acc_div_loss / len(dataloader))
        oth_losses.append(acc_oth_loss / len(dataloader))

    return flow, losses, div_losses, oth_losses

def make_plot_folder(model):
    if not os.path.exists(f'./results/{model}'):
        os.makedirs(f'./results/{model}')
    root_path = f'./results/{model}'
    return root_path

def save_meta_data(args, losses):
    import json
    import pickle
    meta_data = {
        'args': vars(args),
    }
    with open(f'./results/{args.model}/meta_data.json', 'w') as f:
        json.dump(meta_data, f, indent=4)
    with open(f'./results/{args.model}/losses.pkl', 'wb') as f:
        pickle.dump(losses, f)

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='two_peaks')
    parser.add_argument('--num_epochs', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dim_hidden', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--num_kernels', '-k', type=int, default=4)
    parser.add_argument('--non_kernel', '-n', type=bool, default=False)
    args = parser.parse_args()

    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    score_model, dataset = load_model(args.model)
    device = torch.device(args.device)
    flow, losses = train_dynamics(score_model, dataset, **vars(args))
    torch.save(flow.state_dict(), f'./results/{args.model}/dynamics_model.pth')

    root_path = make_plot_folder(args.model)

    save_meta_data(args, losses)

    plt.plot(losses['total'], label='total loss')
    plt.semilogy()
    plt.legend()
    plt.savefig(os.path.join(root_path, 'dynamics_loss.png'))
    plt.savefig(os.path.join(root_path, 'dynamics_loss.pdf'))
    plt.close()

    plot_flow(flow, score_model, DataLoader(dataset, batch_size=256, shuffle=True), device)
    plt.savefig(os.path.join(root_path, 'dynamics_flow.png'))
    plt.savefig(os.path.join(root_path, 'dynamics_flow.pdf'))
    plt.close()