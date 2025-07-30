import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
import wandb
from diffusers import UNet2DModel


class Diffuser:
    def __init__(self, alpha:float):
        self.alpha = torch.tensor(alpha)

    def add_noise(self, x:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        return self.alpha.sqrt() * x + (1 - self.alpha).sqrt() * noise
    
    def estimate_x0(self, xt:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return (xt - (1 - self.alpha).sqrt() * eps) / self.alpha.sqrt()


class ResNet2DBlock(nn.Module):
    """
    A residual block with a customizable kernel size and circular padding.
    The kernel size is determined by `2*r + 1`, and the padding is `r` to
    preserve the input dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int, r: int = 1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            r (int): Radius parameter to control kernel size.
        """
        super(ResNet2DBlock, self).__init__()
        kernel_size = 2 * r + 1
        padding = r

        # The main path of the residual block
        self.main_path = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode='circular',
            ),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode='circular',
            ),
            nn.BatchNorm2d(out_channels)
        )

        # The shortcut connection to match dimensions if necessary
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1, # 1x1 conv to match channel depth
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the residual block."""
        # Add the shortcut (identity) to the output of the main path
        out = self.main_path(x) + self.shortcut(x)
        return F.relu(out)

class ResNet2D(nn.Module):
    """
    A special ResNet-like model for feature extraction where each block has a
    custom number of channels and a custom 'r' value. The final output has the
    same number of channels as the input.
    """
    def __init__(self,
                 in_channels: int,
                 block_channels: Optional[List[int]] = None,
                 block_rs: Optional[List[int]] = None,
                 initial_r: int = 1):
        """
        Args:
            in_channels (int): Number of channels in the input and output tensor.
            block_channels (list of ints, optional): List of hidden channels for each block.
            block_rs (list of ints, optional): List of 'r' values for each block.
            initial_r (int): 'r' value for the initial convolution layer.
        """
        super(ResNet2D, self).__init__()

        # Set default block configurations if none are provided
        if block_channels is None: block_channels = [16, 32, 64, 128]
        if block_rs is None: block_rs = [1, 1, 2, 2]

        # Ensure configuration lists are valid
        if len(block_channels) != len(block_rs):
            raise ValueError("block_channels and block_rs must have the same length.")
        if not block_channels:
            raise ValueError("block_channels cannot be empty.")

        # --- Initial Convolution Layer ---
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels, block_channels[0], kernel_size=2 * initial_r + 1,
                      padding=initial_r, padding_mode='circular'),
            nn.BatchNorm2d(block_channels[0]),
            nn.ReLU(inplace=True)
        )

        # --- Residual Blocks ---
        all_in_channels = [block_channels[0]] + block_channels[:-1]
        self.blocks = nn.Sequential(
            *[ResNet2DBlock(in_c, out_c, r)
              for in_c, out_c, r in zip(all_in_channels, block_channels, block_rs)]
        )

        # --- Final Projection Layer ---
        # A 1x1 convolution to map the final block's channels back to the input channel count.
        self.final_layer = nn.Conv2d(
            block_channels[-1],
            in_channels,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The complete forward pass for the network."""
        x = self.initial_layer(x)
        x = self.blocks(x)
        x = self.final_layer(x)
        return x


class UNet2D(UNet2DModel):
    def forward(self, sample) -> torch.Tensor:
        output = super().forward(sample, timestep=torch.zeros(sample.size(0), device=sample.device))
        return output.sample


class ScoreTrainer:
    def __init__(self, 
            model: nn.Module, 
            diffuser: Diffuser, 
            dataset: torch.utils.data.Dataset, 
            epochs: int = 10, batch_size: int = 32, 
            learning_rate: float = 1e-4, weight_decay: float = 1e-5, 
            device: str = 'cuda', 
            validation_split: float = 0.1, 
            checkpoint_path: str = './model/', 
            log_rate: int = 10,
            num_checkpoints: int = 5,
            warmup_steps: int = 500,
            plot_channel: int = 0,
            use_wandb: bool = False):
        """
        Initializes the Trainer with a model, optimizer, and loss function.
        
        Args:
            model (nn.Module): The neural network model to train.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            loss_fn (nn.Module): The loss function to minimize.
        """
        self.model = model
        self.diffuser = diffuser
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.warmup_steps = warmup_steps
        self.validation_split = validation_split
        self.checkpoint_path = checkpoint_path
        self.use_wandb = use_wandb
        self.log_rate = log_rate
        self.num_checkpoints = num_checkpoints
        self.plot_channel = plot_channel

        # prepare training
        self.train_loader, self.val_loader = self._create_data_loaders()
        self.model.to(self.device)
        self.wandb_setup()
    
    @property
    def config(self):
        c = {}
        for k, v in self.__dict__.items():
            if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                c[k] = v
            elif isinstance(v, bool):
                c[k] = v
        return c
    
    def wandb_setup(self):
        """
        Sets up Weights & Biases for experiment tracking if enabled.
        """
        if self.use_wandb:
            import wandb
            wandb.init(project="score_trainer", config=self.config)
    
    def log(self, metrics: dict):
        """
        Logs metrics to Weights & Biases if enabled.
        
        Args:
            metrics (dict): Dictionary of metrics to log.
        """
        if self.use_wandb:
            wandb.log(metrics)
    
    def _create_data_loaders(self):
        """
        Splits the dataset into training and validation sets and creates data loaders.
        
        Returns:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
        """
        dataset_size = len(self.dataset)
        val_size = int(self.validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        noise = torch.randn_like(x).to(self.device)
        xt = self.diffuser.add_noise(x, noise)
        eps = self.model(xt)
        loss = nn.MSELoss()(eps, noise)
        return loss
    
    def _train_epoch(self):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Training]")

        mse_loss = nn.MSELoss()

        for i, x in enumerate(progress_bar):
            self.scheduler.step()  # Update learning rate
            loss = self.compute_loss(x)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

            if i % self.log_rate == 0:
                self.log({
                    'train_loss': loss.item(), 
                    'epoch': self.current_epoch + i / len(self.train_loader), 
                    'lr': self.optimizer.param_groups[0]['lr']})
    
    def _validate_epoch(self):
        self.model.eval()
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Validation]")

        mse_loss = nn.MSELoss()
        total_loss = 0.0
        with torch.no_grad():
            for i, x in enumerate(progress_bar):
                loss = self.compute_loss(x)
                total_loss += loss.item()

                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.val_loader)
        metrics = {'val_loss': avg_loss, 'epoch': self.current_epoch + 1}
        self.log(metrics)
    
    def train(self):
        """
        Main training loop for the model.
        """
        self.current_epoch = 0
        save_every = self.epochs // self.num_checkpoints if self.num_checkpoints > 0 else 1

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.epochs * len(self.train_loader)
        )

         # Ensure the model is in training mode
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self._train_epoch()
            self._validate_epoch()

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.visualize()
                self.save()
        
        self.save(final=True)
        if self.use_wandb:
            wandb.finish()
    
    def save(self, final: bool = False):
        import os
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        # save model
        if final:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, 'model.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f'model_epoch_{self.current_epoch}.pth'))
        # save config as JSON
        with open(os.path.join(self.checkpoint_path, 'config.json'), 'w') as f:
            import json
            json.dump(self.config, f, indent=4)
        print(f"Model and config saved to {self.checkpoint_path}")
    
    @torch.no_grad()
    def visualize(self):
        self.model.eval()
        idx = random.randint(0, len(self.dataset) - 1)  # random selection
        x = self.dataset[idx].unsqueeze(0).to('cuda')
        x_t = self.diffuser.add_noise(x, torch.randn_like(x))

        eps = self.model(x_t)
        x0_est = self.diffuser.estimate_x0(x_t, eps)

        fig, ax = plt.subplots(1, 3, dpi=300, figsize=(10, 5))

        ax[0].imshow(x[0, self.plot_channel].cpu())
        ax[0].set_title("Original")

        ax[1].imshow(x_t[0, self.plot_channel].cpu())
        ax[1].set_title("Noisy")

        ax[2].imshow(x0_est[0, self.plot_channel].cpu())
        ax[2].set_title("Denoised")

        if self.use_wandb:
            wandb.log({"visualization": fig})
            plt.close()