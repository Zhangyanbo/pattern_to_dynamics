import torch
import torch.nn as nn
from torch.utils.data import Dataset


class GameOfLife(nn.Module):
    """
    Fast Game of Life simulator using PyTorch's Conv2d for efficient computation.
    """
    def __init__(self, device='cpu', padding_mode='zeros'):
        super(GameOfLife, self).__init__()
        self.device = device
        self.padding_mode = padding_mode
        
        # Define the convolution kernel for counting neighbors
        kernel = torch.tensor([
            [1., 1., 1.],
            [1., 0., 1.],
            [1., 1., 1.]
        ], device=self.device).reshape(1, 1, 3, 3)
        
        # Create a fixed convolution layer with the neighbor-counting kernel
        # Use circular padding for toroidal boundary conditions
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, padding_mode=self.padding_mode)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False
        self.to(device)
    
    def step(self, state):
        """
        Compute a single step of Game of Life evolution.
        
        Args:
            state: Tensor of shape (batch_size, 1, height, width) with 0s and 1s
            
        Returns:
            Next state after one step of evolution
        """
        # Count neighbors for each cell
        neighbor_count = self.conv(state)
        return self._apply_rules(state, neighbor_count)
    
    def _apply_rules(self, state, neighbor_count):
        """
        Apply Game of Life rules based on neighbor counts.

        1. Any live cell with 2 or 3 live neighbors survives.
        2. Any dead cell with exactly 3 live neighbors becomes alive.
        3. All other cells die or stay dead.
        """
        survival = state * ((neighbor_count == 2) | (neighbor_count == 3))
        birth = (1 - state) * (neighbor_count == 3)

        return survival + birth
    
    def forward(self, state, steps=1):
        """
        Evolve the Game of Life for multiple steps.
        
        Args:
            state: Initial state tensor of shape (batch_size, 1, height, width)
            steps: Number of evolution steps to compute
            
        Returns:
            Tensor containing all states including the initial one,
            with shape (batch_size, steps+1, 1, height, width)
        """
        states = [state]
        current_state = state
        
        for _ in range(steps):
            current_state = self.step(current_state)
            states.append(current_state)
        
        return torch.stack(states, dim=1)

def create_random_state(batch_size, height, width, density=0.5, device='cpu'):
    """
    Create a random initial state for the Game of Life.
    
    Args:
        batch_size: Number of independent game boards
        height: Height of the game board
        width: Width of the game board
        density: Probability of a cell being alive initially
        device: Device to create the tensor on
        
    Returns:
        Random initial state tensor of shape (batch_size, 1, height, width)
    """
    # Create random tensor with values between 0 and 1
    random_tensor = torch.rand(batch_size, 1, height, width, device=device)
    
    # Convert to binary based on density
    return (random_tensor < density).float()


class GoLDataset(Dataset):
    """
    Game-of-Life snapshots for diffusion / score-matching.

    Each sample: tensor[1, H, W], take the final state after `steps` evolution.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        height: int,
        width: int,
        steps: int = 1,
        density: float = 0.5,
        device: str = "cpu",
        mode: str = "precompute",          # "precompute" | "online"
        chunk: int = 512,                  # batch size for precompute, control peak memory/GPU
        normalize: bool = True,            # Only used in precompute mode
    ):
        super().__init__()
        self.num_samples, self.steps = num_samples, steps
        self.height, self.width = height, width
        self.density, self.device = density, device
        self.mode = mode
        self.sim = GameOfLife(device)      # reuse the same simulator instance

        if mode == "precompute":
            self.data = self.normalize(self._precompute(chunk)) if normalize else self._precompute(chunk)
            self.std = 1; self.mu = 0
        elif mode != "online":
            raise ValueError("mode must be 'precompute' or 'online'")

    # ---------- Dataset API ----------
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == "precompute":
            return self.data[idx]
        else:                               # online
            return self._simulate(1).squeeze(0)

    # ---------- helpers ----------
    def normalize(self, x):
        mu = x.reshape(-1).mean()
        std = x.reshape(-1).std()
        self.mu, self.std = mu, std
        return (x - mu) / std
    
    def denormalize(self, x):
        return x * self.std + self.mu

    @torch.no_grad()
    def _simulate(self, batch_size: int) -> torch.Tensor:
        x0 = create_random_state(
            batch_size, self.height, self.width,
            density=self.density, device=self.device
        )
        xT = self.sim.forward(x0, steps=self.steps)[:, -1]  # take the last step
        return xT                                            # shape (B,1,H,W)

    def _precompute(self, chunk: int) -> torch.Tensor:
        """Generate samples in chunks to avoid memory/GPU overflow"""
        samples = []
        remaining = self.num_samples
        while remaining:
            cur = min(chunk, remaining)
            samples.append(self._simulate(cur).cpu())        # move to CPU to save GPU memory
            remaining -= cur
        return torch.cat(samples, dim=0)                     # shape (N,1,H,W)
