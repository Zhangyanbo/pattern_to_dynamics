from turing_pattern import GrayScottSimulator, create_random_state, TuringPatternDataset
from probflow import Diffuser, ResNet2D, ScoreTrainer
import torch


def train_diffusion(model:str, alpha, trainer_config:dict):
    dataset = TuringPatternDataset.load(f'./turing_pattern/data/{model}_128x128.pt')
    model = ResNet2D(in_channels=2, block_channels=[16, 32, 64], block_rs=[1, 1, 2])
    diffuser = Diffuser(alpha=alpha)

    trainer = ScoreTrainer(
        model=model, 
        diffuser=diffuser, 
        dataset=dataset, 
        **trainer_config
    )

    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train diffusion model on Turing patterns")
    parser.add_argument("-m", type=str, default='waves', help="Model type")
    parser.add_argument("-n", type=int, default=30, help="Number of training epochs")
    parser.add_argument("-b", type=int, default=64, help="Batch size for training")
    parser.add_argument("-l", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--wb", action='store_true', help="Use wandb for training")
    parser.add_argument("-a", type=float, default=0.8, help="Alpha value for diffusion model")
    parser.add_argument('--warmup', type=int, default=500, help="Number of warmup steps for learning rate scheduler")

    args = parser.parse_args()

    trainer_config = dict(
        epochs=args.n, 
        batch_size=args.b, 
        learning_rate=args.l, 
        weight_decay=1e-5, 
        device='cuda', 
        validation_split=0.1, 
        checkpoint_path=f'./turing_pattern/score_models/{args.m}/', 
        log_rate=10,
        num_checkpoints=10,
        warmup_steps=args.warmup,
        use_wandb=args.wb,
    )

    train_diffusion(args.m, args.a, trainer_config)