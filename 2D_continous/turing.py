from turing_pattern import GrayScottSimulator, create_random_state, TuringPatternDataset
from probflow import Diffuser, ResNet2D, ScoreTrainer, UNet2D, DiffusionTrainer, generate_images
import torch, random
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler
from circular import UNet2DModelWithPadding


def get_unet_score():
    unet = UNet2DModelWithPadding(
        # I/O -------------------------------------------------------------
        sample_size=128,          # Board 64x64; only needs to be multiple of 2**(levels-1)
        in_channels=2,           # Game-of-Life: single channel
        out_channels=2,

        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types  =("AttnUpBlock2D",   "UpBlock2D",   "UpBlock2D"),
        block_out_channels=(16, 32, 64), # Channels per layer (>30x smaller than default)

        norm_num_groups=8,  # Reduced GroupNorm groups while maintaining stability
        padding_mode='circular',
        only_when_effective=True,
        log_changed=True
    )
    return unet

def get_unet_diffusion():
    unet = UNet2DModelWithPadding(
        sample_size=128,          # Board 64x64; only needs to be multiple of 2**(levels-1)
        in_channels=2,           # Game-of-Life: single channel
        out_channels=2,

        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types  =("UpBlock2D",   "UpBlock2D",   "UpBlock2D"),
        block_out_channels=(16, 32, 64), # Channels per layer (>30x smaller than default)

        norm_num_groups=8,  # Reduced GroupNorm groups while maintaining stability
        padding_mode='circular',
        only_when_effective=True,
        log_changed=True
    )
    return unet


def train_score(model:str, alpha, trainer_config:dict, network:str='unet'):
    dataset = TuringPatternDataset.load(f'./turing_pattern/data/{model}_128x128.pt')
    if network == 'unet':
        model = get_unet_score()
    else:
        model = ResNet2D(in_channels=2, block_channels=[16, 32, 64], block_rs=[1, 1, 2])
    diffuser = Diffuser(alpha=alpha)

    trainer = ScoreTrainer(
        model=model, 
        diffuser=diffuser, 
        dataset=dataset, 
        **trainer_config
    )

    trainer.train()

def train_diffusion(model_name:str, alpha, trainer_config:dict, network:str='unet'):
    dataset = TuringPatternDataset.load(f'./turing_pattern/data/{model_name}_128x128.pt')
    if network == 'unet':
        model = get_unet_diffusion()
    else:
        raise ValueError(f"Unsupported network type for training diffusion models: {network}")
    scheduler = DDPMScheduler(
        num_train_timesteps=100, 
        beta_start=0.0001, beta_end=0.1,
        prediction_type="epsilon"
    )

    trainer = DiffusionTrainer(
        unet=model,
        model_name=model_name,
        scheduler=scheduler,
        dataset=dataset,
        **trainer_config
    )

    trainer.train()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train diffusion model on Turing patterns")
    parser.add_argument("--model", "-m", type=str, default='waves', help="Model type")
    parser.add_argument("--network", help="Network type", choices=['resnet', 'unet'], default='unet')
    parser.add_argument("--objective", choices=['score', 'diffusion'], default='score', help="Objective to optimize")
    parser.add_argument("--epoch", "-n", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", '-b', type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", '-l', type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--wb", action='store_true', help="Use wandb for training")
    parser.add_argument("--alpha", "-a", type=float, default=0.8, help="Alpha value for score model")
    parser.add_argument('--warmup', type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument('--lr_schedule', action='store_true', help="Use cosine learning rate schedule")
    parser.add_argument('--plot_channel', type=int, default=0, help="Channel to plot during training")
    parser.add_argument('--sampling', choices=['ddpm', 'ddim'], default='ddpm', help="Sampling method for diffusion model")
    parser.add_argument('--ema', action='store_true', help="Use EMA for diffusion model")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.objective == 'score':
        trainer_config = dict(
            epochs=args.epoch, 
            batch_size=args.batch, 
            learning_rate=args.lr, 
            weight_decay=1e-5, 
            device='cuda', 
            validation_split=0.1, 
            checkpoint_path=f'./turing_pattern/score_models/{args.model}/', 
            log_rate=10,
            num_checkpoints=10,
            warmup_steps=args.warmup,
            use_wandb=args.wb,
            plot_channel=args.plot_channel,
        )

        train_score(args.model, args.alpha, trainer_config, network=args.network)
    elif args.objective == 'diffusion':
        trainer_config = dict(
            epochs=args.epoch, 
            batch_size=args.batch, 
            learning_rate=args.lr, 
            task_name='turing_diffusion',
            weight_decay=0.01, 
            device='cuda', 
            validation_split=0.1, 
            checkpoint_path=f'./turing_pattern/diffusion_models/{args.model}/', 
            warmup_steps=args.warmup,
            use_wandb=args.wb,
            method=args.sampling,
            lr_schedule=args.lr_schedule,
            use_ema=args.ema,
        )

        train_diffusion(args.model, args.alpha, trainer_config, network=args.network)