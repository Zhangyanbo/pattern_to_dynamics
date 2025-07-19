import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb
from transformers import get_cosine_schedule_with_warmup
from diffusers.training_utils import compute_snr


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
    def __init__(self, unet: UNet2DModel, scheduler: DDPMScheduler, dataset: torch.utils.data.Dataset,
                 epochs: int = 100, learning_rate: float = 1e-4, batch_size: int = 32, weight_decay: float = 0.01,
                 task_name: str = 'diffusion', use_wandb: bool = False, warmup_steps: int = 500, gamma: float = 5,
                 device: str = 'cuda', validation_split: float = 0.1, checkpoint_path: str = 'unet_checkpoint.pth'):

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
        self.gamma = gamma

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

    def _train_epoch(self):
        """Runs a single training epoch."""
        self.unet.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Training]")

        i = 0
        for batch in progress_bar:
            self.optimizer_scheduler.step()
            
            clean_images = batch.to(self.device)
            
            # 1. Sample noise
            noise = torch.randn_like(clean_images)
            
            # 2. Sample a random timestep for each image
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=self.device).long()
            snr = compute_snr(self.scheduler, timesteps)
            weights = torch.clamp(snr, max=self.gamma) / snr
            
            # 3. Add noise to the clean images
            noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
            
            # --- Forward Pass ---
            self.optimizer.zero_grad()
            noise_pred = self.unet(noisy_images, timesteps, return_dict=False)[0]
            
            # --- Loss Calculation ---
            base_loss = (noise_pred - noise).pow(2).mean(dim=(1,2,3))
            loss = (base_loss * weights).mean()
            
            # --- Backward Pass & Optimization ---
            loss.backward()
            # clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
            self.log({'train_loss': base_loss.mean().item(), 'epoch': self.current_epoch + i / len(self.train_loader), 'lr': self.optimizer.param_groups[0]['lr']})
            self.optimizer.step()

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
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=self.device).long()
                snr = compute_snr(self.scheduler, timesteps)
                weights = torch.clamp(snr, max=self.gamma) / snr
                noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
                
                noise_pred = self.unet(noisy_images, timesteps, return_dict=False)[0]
                base_loss = (noise_pred - noise).pow(2).mean(dim=(1,2,3))
                loss = (base_loss * weights).mean()

                total_loss += base_loss.mean().item()
                progress_bar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = total_loss / len(self.val_loader)
        self.log({'val_loss': avg_val_loss, 'epoch': self.current_epoch + 1})
        self.history['val_loss'].append(avg_val_loss)
        return avg_val_loss
    
    def log_generated_images(self):
        if self.use_wandb:
            # Generate a batch of images for logging
            # shape = (1, 2, 128, 128)
            generated_images = generate_images(
                self.unet, self.scheduler, 
                height=128, width=128, batch_size=1, 
                num_inference_steps=1000, device=self.device
            )
            # convert to a format suitable for logging
            w = generated_images[0]
            u = (w[0] - w[0].min()) / (w[0].max() - w[0].min() + 1e-3)
            v = (w[1] - w[1].min()) / (w[1].max() - w[1].min() + 1e-3)
            img = torch.stack([u, v, (u + v) / 2])
            # Log the generated images to wandb
            image = wandb.Image(img.cpu(), caption="Generated Image")
            wandb.log({"generated_image": image})

    def train(self):
        """The main training loop."""
        print("Starting training process...")
        self._split_dataset()
        self._create_dataloaders()

        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.optimizer_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.epochs * len(self.train_loader)
        )

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            
            avg_train_loss = self._train_epoch()
            avg_val_loss = self._validate_epoch()

            self.logger.step()
            
            print(f"Epoch {epoch}/{self.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if (epoch + 1) % int(self.epochs / 10) == 0:
                print(f"Saving checkpoint for epoch {epoch}...")
                self.log_generated_images()
                self.save_model()

        print("Training finished.")
        self.save_model()

    def save_model(self):
        """Saves the trained UNet model state dictionary."""
        torch.save(self.unet.state_dict(), self.checkpoint_path)
        print(f"Model checkpoint saved to {self.checkpoint_path}")

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
    num_inference_steps: int = 1000,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Generates images by running the full reverse diffusion process.

    Args:
        unet (UNet2DModel): The trained UNet model to use for denoising.
        scheduler (DDPMScheduler): The noise scheduler.
        image_size (int): The size of the square images to generate.
        batch_size (int): The number of images to generate in a batch.
        num_inference_steps (int): The number of denoising steps.
        device (str): The device to run the generation on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor containing the generated images.
    """
    print("Starting image generation...")

    # --- Device Setup ---
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = 'cpu'
    unet.to(device)
    unet.eval() # Set the model to evaluation mode

    # --- Initial Noise ---
    # Start with random noise that will be progressively denoised.
    # The shape is (batch_size, num_channels, height, width)
    image = torch.randn(
        (batch_size, unet.config.in_channels, height, width),
        device=device,
    )

    # --- Set Timesteps ---
    # The scheduler will guide the denoising process over these timesteps.
    scheduler.set_timesteps(num_inference_steps)

    # --- Denoising Loop ---
    progress_bar = tqdm(scheduler.timesteps, desc="Generating Images")
    for t in progress_bar:
        with torch.no_grad():
            # 1. Predict the noise residual for the current timestep
            noise_pred = unet(image, t, return_dict=False)[0]

        # 2. Compute the previous noisy sample using the scheduler's step method
        # This "denoises" the image by a small amount
        image = scheduler.step(noise_pred, t, image, return_dict=False)[0]

    return image