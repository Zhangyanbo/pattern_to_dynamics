import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDPMPipeline
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb
from transformers import get_cosine_schedule_with_warmup
import inspect
from diffusers.training_utils import EMAModel
from typing import Tuple


class Logger:
    """
    A logger for tracking and averaging values over time.

    This class maintains a buffer of values and computes their average over time.

    Example Usage:
    ```python
    logger = Logger()
    for epoch in range(10):
        for i in range(1000):
            logger.log('loss', i)
        logger.step()
    print(logger['loss'])
    ```
    """

    def __init__(self):
        self.buffer = {}
        self.history = {}

    def log(self, name, value):
        if name not in self.buffer:
            self.buffer[name] = []
        self.buffer[name].append(value)

    def step(self):
        for name, values in self.buffer.items():
            if name not in self.history:
                self.history[name] = []
            if values:
                avg_value = sum(values) / len(values)
                self.history[name].append(avg_value)
        self.buffer.clear()

    def __getitem__(self, key):
        return self.history.get(key, [])


class DiffusionTrainer:
    """
    A trainer class for training a UNet model for a diffusion process
    using the Hugging Face Diffusers library.

    Args:
        unet (UNet2DModel): The UNet model to be trained.
        scheduler (DDPMScheduler): The noise scheduler.
        dataset (torch.utils.data.Dataset): The dataset of images.
        epochs (int): Number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The size of each training batch.
        device (str): The device to train on ('cuda' or 'cpu').
        validation_split (float): The fraction of the dataset to use for validation.
        checkpoint_path (str): Path to save the trained model checkpoint.
    """
    def __init__(self, 
                 unet: UNet2DModel, 
                 scheduler: DDPMScheduler, 
                 dataset: torch.utils.data.Dataset,
                 model_name:str = 'diffusion_model',
                 epochs: int = 100, 
                 learning_rate: float = 1e-4, 
                 batch_size: int = 32, 
                 weight_decay: float = 0.01,
                 task_name: str = 'diffusion', 
                 use_wandb: bool = False, 
                 warmup_steps: int = 500, 
                 device: str = 'cuda', 
                 validation_split: float = 0.1, 
                 checkpoint_path: str = 'unet_checkpoint.pth',
                 method:str = 'ddim',
                 clip_sample_range:float = 1,
                 lr_schedule:bool = False,
                 use_ema:bool = False
                 ):

        # --- Initialization ---
        self.unet = unet
        self.scheduler = scheduler
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.checkpoint_path = checkpoint_path
        self.logger = Logger()
        self.use_wandb = use_wandb
        self.task_name = task_name
        self.warmup_steps = warmup_steps
        self.method = method
        self.clip_sample_range = clip_sample_range
        self.lr_schedule = lr_schedule
        self.use_ema = use_ema
        self.model_name = model_name

        # --- Device Setup ---
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            self.device = 'cpu'
        else:
            self.device = device
        self.unet.to(self.device)

        # --- WandB Setup ---
        self.wandb_setup()

        # --- Loss History ---
        self.history = {'train_loss': [], 'val_loss': []}
    
    def log(self, info:dict):
        for name, value in info.items():
            self.logger.log(name, value)
        if self.use_wandb:
            wandb.log(info)
    
    def wandb_setup(self):
        if self.use_wandb:
            run = wandb.init(
                    project=self.task_name,
                    group=self.model_name,
                    # Track hyperparameters and run metadata.
                    config=dict(
                        learning_rate=self.learning_rate,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_split=self.validation_split,
                        device=self.device,
                        checkpoint_path=self.checkpoint_path,
                        warmup_steps=self.warmup_steps,
                        weight_decay=self.weight_decay,
                        task_name=self.task_name
                    ),
                )

    def _split_dataset(self):
        """Splits the dataset into training and validation sets."""
        dataset_size = len(self.dataset)
        val_size = int(np.floor(self.validation_split * dataset_size))
        train_size = dataset_size - val_size
        
        print(f"Dataset size: {dataset_size}")
        print(f"Training set size: {train_size}")
        print(f"Validation set size: {val_size}")

        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def _create_dataloaders(self):
        """Creates DataLoader instances for training and validation."""
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def get_loss(self, clean_images):
        lf = nn.MSELoss()
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=self.device).long()
        
        # 3. Add noise to the clean images
        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
        
        # --- Forward Pass ---
        noise_pred = self.unet(noisy_images, timesteps, return_dict=False)[0]
        
        # --- Loss Calculation ---
        if self.scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(clean_images, noise, timesteps)
        loss = lf(noise_pred, target)
        return loss

    def _train_epoch(self):
        """Runs a single training epoch."""
        self.unet.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Training]")

        i = 0
        for batch in progress_bar:
            clean_images = batch.to(self.device)

            self.optimizer.zero_grad()
            loss = self.get_loss(clean_images)
            loss.backward()

            # clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
            self.log({
                'train_loss': loss.mean().item(), 
                'epoch': self.current_epoch + i / len(self.train_loader), 
                'lr': self.optimizer.param_groups[0]['lr']
                })
            self.optimizer.step()
            if self.lr_schedule:
                self.optimizer_scheduler.step()
            if self.use_ema:
                self.ema_unet.step(self.unet.parameters())

            i += 1
            
            total_loss += loss.item()
            progress_bar.set_postfix({'train_loss': loss.item()})
            
        avg_train_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_train_loss)
        return avg_train_loss

    def _validate_epoch(self):
        """Runs a single validation epoch."""
        self.unet.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Validation]")

        with torch.no_grad():
            for batch in progress_bar:
                clean_images = batch.to(self.device)
                loss = self.get_loss(clean_images)

                total_loss += loss.item()
                progress_bar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = total_loss / len(self.val_loader)
        self.log({'val_loss': avg_val_loss, 'epoch': self.current_epoch + 1})
        self.history['val_loss'].append(avg_val_loss)
        return avg_val_loss
    
    @staticmethod
    def _normalize_image(w):
        if isinstance(w, np.ndarray):
            w = torch.from_numpy(w).float()
        u0 = w[0]
        v0 = w[1]
        # Normalize the image to [0, 1] range
        u = (u0 - u0.min()) / (u0.max() - u0.min() + 1e-3)
        v = (v0 - v0.min()) / (v0.max() - v0.min() + 1e-3)
        img = torch.stack([u, v, (u + v) / 2])
        return img
    
    def log_generated_images(self):
        images = generate_images(
            self.unet, 
            self.scheduler,
            height=self.unet.config.sample_size, 
            width=self.unet.config.sample_size, 
            batch_size=1,
            eta=0.0,
            num_inference_steps=self.scheduler.config['num_train_timesteps'], 
            device=self.device
        ).cpu().clamp(-self.clip_sample_range, self.clip_sample_range)

        num_channels = images.shape[1]
        idx = np.random.randint(0, len(self.dataset))

        # convert to a format suitable for logging
        if num_channels == 2:
            img = self._normalize_image(images[0])
            img_real = self._normalize_image(self.dataset[idx])
        elif num_channels == 3:
            img = self.dataset.denormalize(images[0])
            img_real = self.dataset.denormalize(self.dataset[idx])

        # Make plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))

        ax[0].imshow(img_real.permute(1, 2, 0).cpu().numpy())
        ax[0].axis('off')
        ax[0].set_title("Real Image")
        ax[1].imshow(img.permute(1, 2, 0).cpu().numpy())
        ax[1].axis('off')
        ax[1].set_title(f"Generated Image at Epoch {self.current_epoch}")

        fig.tight_layout()
        if self.use_wandb:
            # Log the generated images to wandb
            wandb.log({"generated_image": fig})
        else:
            # save the image using matplotlib
            plt.savefig(f"sampled.png", bbox_inches='tight')
            plt.close()

    def train(self):
        """The main training loop."""
        print("Starting training process...")
        self._split_dataset()
        self._create_dataloaders()

        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate, 
                                          weight_decay=self.weight_decay)
        if self.lr_schedule:
            self.optimizer_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.epochs * len(self.train_loader)
            )
        
        if self.use_ema:
            self.ema_unet = EMAModel(
                        parameters=self.unet.parameters(),
                        decay=0.9999,
                        model_cls=type(self.unet),
                        model_config=self.unet.config,
                    )

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            
            avg_train_loss = self._train_epoch()

            if self.use_ema:
                self.ema_unet.store(self.unet.parameters())
                self.ema_unet.copy_to(self.unet.parameters())
            avg_val_loss = self._validate_epoch()

            if (epoch + 1) % max(1, int(self.epochs / 10)) == 0:
                print(f"Saving checkpoint for epoch {epoch}...")
                self.log_generated_images()
                self.save_model()

            if self.use_ema:
                self.ema_unet.restore(self.unet.parameters())

            self.logger.step()
            
            print(f"Epoch {epoch}/{self.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        print("Training finished.")
        self.save_model()

    def save_model(self):
        """Saves the trained UNet model state dictionary."""
        if self.use_ema:
            self.ema_unet.save_pretrained(self.checkpoint_path)
        else:
            self.unet.save_pretrained(self.checkpoint_path)
        
        self.scheduler.save_pretrained(self.checkpoint_path)
        print(f"Model saved to {self.checkpoint_path}")

    def plot_loss_curves(self):
        """Plots the training and validation loss curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


def generate_images(
    unet,
    scheduler,
    height: int = 128,
    width: int = 128,
    batch_size: int = 4,
    num_inference_steps: int = 100,
    device: str = "cuda",
    eta: float = 0.0, 
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    unet.to(device).eval()

    img = torch.randn(
        (batch_size, unet.config.in_channels, height, width),
        device=device,
    )

    if isinstance(scheduler, DDIMScheduler):
        arguments = {'eta': eta}
    else:
        arguments = {}

    scheduler.set_timesteps(num_inference_steps, device=device)

    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        with torch.no_grad():
            model_output = unet(img, t).sample
        img = scheduler.step(model_output, t, img, **arguments).prev_sample

    return img