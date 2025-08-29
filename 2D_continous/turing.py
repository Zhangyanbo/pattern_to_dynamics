from turing_pattern import TuringPatternDataset
from probflow import DiffusionTrainer
import torch, random
import numpy as np
from diffusers import DDPMScheduler
from circular import UNet2DModelWithPadding


def get_unet_diffusion():
    unet = UNet2DModelWithPadding(
        sample_size=128,
        in_channels=2,
        out_channels=2,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types  =("UpBlock2D",   "UpBlock2D",   "UpBlock2D"),
        block_out_channels=(16, 32, 64),
        norm_num_groups=8,
        padding_mode='circular',
        only_when_effective=True,
        log_changed=True
    )
    return unet


def train_diffusion(model_name:str, trainer_config:dict, prediction_type='epsilon'):
    dataset = TuringPatternDataset.load(f'./turing_pattern/data/{model_name}_128x128.pt')
    model = get_unet_diffusion()
    scheduler = DDPMScheduler(
        num_train_timesteps=100, 
        beta_start=0.0001, beta_end=0.1,
        prediction_type=prediction_type
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
    parser.add_argument("--epoch", "-n", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", '-b', type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", '-l', type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--wb", action='store_true', help="Use wandb for training")
    parser.add_argument('--warmup', type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument('--lr_schedule', action='store_true', help="Use cosine learning rate schedule")
    parser.add_argument('--sampling', choices=['ddpm', 'ddim'], default='ddpm', help="Sampling method for diffusion model")
    parser.add_argument('--ema', action='store_true', help="Use EMA for diffusion model")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--prediction_type', type=str, default='epsilon', help="Prediction type for diffusion model")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        use_ema=args.ema
    )

    train_diffusion(args.model, trainer_config, prediction_type=args.prediction_type)