from turing_pattern import GrayScottSimulator, create_random_state, TuringPatternDataset
from diffusers import UNet2DModel, DDPMScheduler
from probflow import DiffusionTrainer, generate_images
import matplotlib.pyplot as plt
import torch


GS_config = dict(
    sample_size=128,
    in_channels=2,
    out_channels=2,

    # Architecture ---------------------------------------------------
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types  =("UpBlock2D",   "UpBlock2D",   "UpBlock2D",  "UpBlock2D"),
    block_out_channels=(16, 32, 64, 128), 
    norm_num_groups=8,
)


def plot_figures(patterns:torch.tensor) -> None:
    for i in range(4):
        w = patterns[i]
        u = (w[0] - w[0].min()) / (w[0].max() - w[0].min() + 1e-3)
        v = (w[1] - w[1].min()) / (w[1].max() - w[1].min() + 1e-3)
        img = torch.stack([u, v, (u + v) / 2])

        plt.subplot(2, 2, i + 1)
        plt.imshow(img.permute(1, 2, 0).squeeze().cpu().numpy())

def plot_loss(trainer) -> None:
    plt.plot(trainer.logger['train_loss'])
    plt.plot(trainer.logger['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.yscale('log')

def train_diffusion(model:str, trainer_config:dict):
    dataset = TuringPatternDataset.load(f'./turing_pattern/data/{model}_128x128.pt')
    unet = UNet2DModel(**GS_config)
    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon", beta_schedule="squaredcos_cap_v2")

    trainer = DiffusionTrainer(
        unet=unet, 
        scheduler=scheduler, 
        dataset=dataset, 
        **trainer_config
    )

    trainer.train()

    sampled = generate_images(
        unet, scheduler, height=128, width=128, batch_size=4, num_inference_steps=1000, device='cuda')
    
    # plot loss
    plot_loss(trainer)
    plt.savefig(f'./turing_pattern/figures/{model}_loss.png')
    plt.close()

    # plot sampled
    plot_figures(sampled)
    plt.savefig(f'./turing_pattern/figures/{model}_generated_patterns.png')
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train diffusion model on Turing patterns")
    parser.add_argument("-m", type=str, default='waves', choices=['waves', 'life'], help="Model type")
    parser.add_argument("-n", type=int, default=30, help="Number of training epochs")
    parser.add_argument("-b", type=int, default=64, help="Batch size for training")
    parser.add_argument("-l", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--wb", action='store_true', help="Use wandb for training")
    parser.add_argument('--warmup', type=int, default=500, help="Number of warmup steps for learning rate scheduler")

    args = parser.parse_args()

    trainer_config = dict(
        epochs=args.n,
        learning_rate=args.l,
        batch_size=args.b,
        device='cuda',
        validation_split=0.05,
        checkpoint_path='./turing_pattern/data/diffusion_checkpoint.pth',
        use_wandb=args.wb,
        warmup_steps=args.warmup,
    )

    train_diffusion(args.m, trainer_config)