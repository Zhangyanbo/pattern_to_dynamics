from diffusers import UNet2DModel, DDPMScheduler
from GoLData import GoLDataset
from tqdm import tqdm
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from plot import plot_gol_comparison


def get_model():
    unet = UNet2DModel(
        # I/O -------------------------------------------------------------
        sample_size=64,          # Board 64x64; only needs to be multiple of 2**(levels-1)
        in_channels=1,           # Game-of-Life: single channel
        out_channels=1,

        # Architecture ---------------------------------------------------
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types  =("UpBlock2D",   "UpBlock2D",   "UpBlock2D"),
        block_out_channels=(16, 32, 64), # Channels per layer (>30x smaller than default)
        layers_per_block=1,              # 1 ResNet per layer; receptive field enough for 3x3 rules
        add_attention=False,             # Disable self-attention
        attention_head_dim=None,         # Avoid wasting parameters

        # Training-friendly hyperparameters -------------------------------
        dropout=0.0,        # Simple task, no regularization needed
        norm_num_groups=8,  # Reduced GroupNorm groups while maintaining stability
    )
    return unet


def train_diffusion(num_epochs=500):
    dataset = GoLDataset(num_samples=1024 * 8, height=64, width=64, steps=128, device='cuda', mode='precompute', normalize=False)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")

    unet = get_model().cuda()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-3, weight_decay=1e-5)
    training_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)


    losses = []
    lf = nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        loss_cum = 0.0
        for x0 in dataloader:
            x0 = x0.to("cuda")                               # (B,1,64,64)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (x0.shape[0],), device=x0.device).long()

            noise   = torch.randn_like(x0)
            x_t     = scheduler.add_noise(x0, noise, t)      # q(x_t|x0)
            eps_pred = unet(x_t, t).sample                   # εθ

            loss = lf(eps_pred, noise)               # L_simple
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_cum += loss.item()
            training_scheduler.step()
        losses.append(loss_cum / len(dataloader))
    unet.eval()
    return unet, losses, x0.cpu(), scheduler

def plot_loss(losses):
    plt.plot(losses)
    plt.savefig('./figures/loss.pdf')
    plt.semilogy()
    plt.close()

def plot_compare(unet, x_gol, scheduler):
    unet.eval()  # Set to evaluation mode
    with torch.no_grad():
        # Start with random noise
        x_t = torch.randn(1, 1, 64, 64).cuda()
        
        # Gradually denoise through timesteps
        for t in reversed(range(scheduler.config.num_train_timesteps)):
            t_tensor = torch.tensor([t], device='cuda')
            noise_pred = unet(x_t, t_tensor).sample
            x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    
    plot_gol_comparison(x_gol[0, 0], x_t[0, 0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'plot'],
                        help='train model or plot samples from saved model')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of epochs to train')
    args = parser.parse_args()

    if args.mode == 'train':
        unet, losses, x_sample, scheduler = train_diffusion(args.num_epochs)
        torch.save(unet.state_dict(), './models/GoL_diffusion.pth')
        with open('./models/GoL_diffusion.json', 'w') as f:
            json.dump(unet.config, f, indent=4)
        plot_loss(losses)
        plot_compare(unet, x_sample, scheduler)
    else:
        # Load model config and create model
        with open('./models/GoL_diffusion.json', 'r') as f:
            config = json.load(f)
        unet = UNet2DModel(**config).cuda()
        unet.load_state_dict(torch.load('./models/GoL_diffusion.pth'))
        
        # Get a random sample from training data for comparison
        dataset = GoLDataset(num_samples=1, height=64, width=64, steps=128, device='cuda', mode='precompute', normalize=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
        x_sample = next(iter(dataloader)).cuda()
        plot_compare(unet, x_sample, scheduler)