from diffusers import UNet1DModel, DDPMScheduler
from ECAData import ECADataset
from tqdm import tqdm
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from plot import plot_eca_sorted


SCHEDULER_STEPS = 100  # Number of diffusion steps
SIMULATION_STEPS = 32  # Number of ECA evolution steps
RULE = 1  # ECA rule number (0-255)
SPACE_SIZE = 32  # Width of ECA patterns

def get_model():
    unet = UNet1DModel(
        # I/O -------------------------------------------------------------
        sample_size=SPACE_SIZE,          # Board 64; only needs to be multiple of 2**(levels-1)
        in_channels=1,           # ECA: single channel
        out_channels=1,
        # time_embedding_type="positional",  # Fourier time embeddings for 1D data

        # Architecture ---------------------------------------------------
        down_block_types=("DownBlock1D", "AttnDownBlock1D"),
        up_block_types  =("AttnUpBlock1D",   "UpBlock1D"),
        block_out_channels=(32, 64), # Channels per layer (>30x smaller than default)
        layers_per_block=1,              # 1 ResNet per layer; receptive field enough for 3x3 rules
    )
    return unet


def train_diffusion(num_epochs=500, num_samples=1024 * 8):
    dataset = ECADataset(rule=RULE, num_samples=num_samples, width=SPACE_SIZE, steps=SIMULATION_STEPS, device='cuda',
                         mode='precompute', normalize=False, padding_mode='circular')
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    scheduler = DDPMScheduler(num_train_timesteps=SCHEDULER_STEPS,
                              prediction_type="epsilon")

    unet = get_model().cuda()
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4, weight_decay=1e-7)
    training_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)


    losses = []
    lf = nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        loss_cum = 0.0
        for x0 in dataloader:
            x0 = x0.to("cuda")                               # (B,1,64,64)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (x0.shape[0],), device=x0.device).long()

            noise   = torch.randn_like(x0)
            x_t     = scheduler.add_noise(x0, noise, t)      # q(x_t|x0)
            eps_pred = unet(x_t, t, return_dict=False)[0]                   # εθ

            loss = lf(eps_pred, noise)               # L_simple
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_cum += loss.item()
            # training_scheduler.step()
        losses.append(loss_cum / len(dataloader))
    unet.eval()
    return unet, losses, x0.cpu(), scheduler

def plot_loss(losses):
    plt.plot(losses)
    plt.semilogy()
    plt.savefig('./figures/loss.pdf')
    plt.close()

def plot_compare(unet, x_eca, scheduler):
    print(f'ECA shape: {x_eca.shape}')
    unet.eval()  # Set to evaluation mode
    with torch.no_grad():
        # Start with random noise
        x_t = torch.randn(128, 1, SPACE_SIZE).cuda()
        
        # Gradually denoise through timesteps
        for t in reversed(range(scheduler.config.num_train_timesteps)):
            t_tensor = torch.tensor([t], device='cuda')
            noise_pred = unet(x_t, t_tensor).sample
            x_t = scheduler.step(noise_pred, t, x_t).prev_sample
        
    print(f'Generated ECA shape: {x_t.shape}')

    plot_eca_sorted(x_eca.squeeze().cpu(), x_t.squeeze().cpu())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'plot'],
                        help='train model or plot samples from saved model')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of epochs to train')
    parser.add_argument('--num_samples', type=int, default=1024 * 8,
                        help='number of samples to train on')
    args = parser.parse_args()

    if args.mode == 'train':
        unet, losses, x_sample, scheduler = train_diffusion(args.num_epochs, args.num_samples)
        torch.save(unet.state_dict(), './models/ECA_diffusion.pth')
        with open('./models/ECA_diffusion.json', 'w') as f:
            json.dump(unet.config, f, indent=4)
        with open('./models/losses.json', 'w') as f:
            json.dump(losses, f, indent=4)
        plot_loss(losses)
        plot_compare(unet, x_sample, scheduler)
    else:
        # Load model config and create model
        with open('./models/ECA_diffusion.json', 'r') as f:
            config = json.load(f)
        with open('./models/losses.json', 'r') as f:
            losses = json.load(f)
        unet = UNet1DModel(**config).cuda()
        unet.load_state_dict(torch.load('./models/ECA_diffusion.pth'))
        
        # Get a random sample from training data for comparison
        dataset = ECADataset(rule=RULE, num_samples=128, width=SPACE_SIZE, steps=SIMULATION_STEPS, device='cuda',
                             mode='precompute', normalize=False, padding_mode='circular')
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        scheduler = DDPMScheduler(
            num_train_timesteps=SCHEDULER_STEPS, prediction_type="epsilon")
        x_sample = next(iter(dataloader)).cuda()
        plot_compare(unet, x_sample, scheduler)
        plot_loss(losses)
