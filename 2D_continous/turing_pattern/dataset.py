import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import os
from tqdm import tqdm


# Gray-Scott Turing Pattern Dataset
# Converted from Game of Life dataset to reaction-diffusion patterns
# Maintains the same API but generates continuous Turing patterns instead of discrete cellular automata


class GrayScottSimulator(nn.Module):
    """
    Gray-Scott reaction-diffusion simulator for generating Turing patterns.
    
    The Gray-Scott model simulates two chemical species U and V:
    ∂u/∂t = Du∇²u - uv² + F(1-u)
    ∂v/∂t = Dv∇²v + uv² - (F+k)v
    
    Where:
    - Du, Dv: Diffusion coefficients (typically Du/Dv ≈ 2)
    - F: Feed rate (rate at which U is replenished)
    - k: Kill rate (rate at which V is removed)
    
    Different parameter combinations produce different patterns:
    - Spots: F=0.035, k=0.065 (leopard-like spots)
    - Stripes: F=0.035, k=0.06 (zebra-like stripes)
    - Spirals: F=0.014, k=0.054 (rotating spirals)
    - Waves: F=0.025, k=0.055 (traveling waves)
    - And many more complex patterns...
    """
    
    def __init__(self, Du=0.16, Dv=0.08, F=0.035, k=0.065, dt=1.0, clamp=True, device='cpu'):
        """
        Args:
            Du: Diffusion rate of U (typically 0.16)
            Dv: Diffusion rate of V (typically 0.08)
            F: Feed rate (typically 0.01-0.1)
            k: Kill rate (typically 0.045-0.07)
            dt: Time step for integration
            device: Computation device
        """
        super(GrayScottSimulator, self).__init__()
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        self.dt = dt
        self.device = device
        self.clamp = clamp
        
        # Laplacian kernel for diffusion
        laplacian_kernel = torch.tensor([
            [0., 1., 0.],
            [1., -4., 1.],
            [0., 1., 0.]
        ], device=device).reshape(1, 1, 3, 3)
        
        # Create convolution layers for computing Laplacian
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, padding_mode='circular')
        self.laplacian.weight.data = laplacian_kernel
        self.laplacian.weight.requires_grad = False
        
        self.to(device)
    
    def compute_laplacian(self, x):
        """Compute discrete Laplacian using convolution."""
        return self.laplacian(x)
    
    def step(self, u, v):
        """
        Compute one time step of the Gray-Scott model.
        
        Args:
            u: Concentration of species U, shape (batch, 1, height, width)
            v: Concentration of species V, shape (batch, 1, height, width)
            
        Returns:
            Tuple of (u_next, v_next) after one time step
        """
        # Compute Laplacians
        lap_u = self.compute_laplacian(u)
        lap_v = self.compute_laplacian(v)
        
        # Reaction terms
        uvv = u * v * v
        
        # Update equations
        du_dt = self.Du * lap_u - uvv + self.F * (1 - u)
        dv_dt = self.Dv * lap_v + uvv - (self.F + self.k) * v
        
        # Euler integration
        u_next = u + self.dt * du_dt
        v_next = v + self.dt * dv_dt
        
        # Clamp to ensure stability
        if self.clamp:
            u_next = torch.clamp(u_next, 0, 1)
            v_next = torch.clamp(v_next, 0, 1)
        
        return u_next, v_next
    
    def forward(self, u, v, steps=1, trace=False):
        """
        Evolve the Gray-Scott model for multiple steps.
        
        Args:
            u: Initial U concentration, shape (batch, 1, height, width)
            v: Initial V concentration, shape (batch, 1, height, width)
            steps: Number of time steps
            
        Returns:
            Tuple of tensors (u_states, v_states), each with shape 
            (batch, steps+1, 1, height, width) containing all states
        """
        u_states = [u]
        v_states = [v]
        
        for _ in range(steps):
            u, v = self.step(u, v)
            if trace:
                u_states.append(u)
                v_states.append(v)
            else:
                u_states[0] = u
                v_states[0] = v
        
        return torch.stack(u_states, dim=1), torch.stack(v_states, dim=1)


def create_random_state(batch_size, height, width, pattern_type='random', device='cpu'):
    """
    Create initial conditions for the Gray-Scott model.
    
    Args:
        batch_size: Number of independent simulations
        height: Height of the domain
        width: Width of the domain
        pattern_type: 'random', 'center_seed', or 'multiple_seeds'
        device: Device to create tensors on
        
    Returns:
        Tuple of (u, v) initial concentrations
    """
    # Base state: U=1, V=0 everywhere
    u = torch.ones(batch_size, 1, height, width, device=device)
    v = torch.zeros(batch_size, 1, height, width, device=device)
    
    if pattern_type == 'random':
        # Add small random perturbations
        u += 0.05 * torch.randn_like(u)
        v += 0.05 * torch.randn_like(v)
        
    elif pattern_type == 'center_seed':
        # Single seed in center
        cy, cx = height // 2, width // 2
        r = min(height, width) // 10
        y, x = torch.meshgrid(torch.arange(height, device=device), 
                              torch.arange(width, device=device), indexing='ij')
        mask = ((y - cy)**2 + (x - cx)**2) < r**2
        u[:, 0, mask] = 0.5
        v[:, 0, mask] = 0.25
        
    elif pattern_type == 'multiple_seeds':
        # Multiple random seeds
        n_seeds = 5
        for b in range(batch_size):
            for _ in range(n_seeds):
                r = torch.randint(3, 8, (1,)).item()  # Define r first
                cy = torch.randint(r, height-r, (1,)).item()
                cx = torch.randint(r, width-r, (1,)).item()
                y, x = torch.meshgrid(torch.arange(height, device=device), 
                                      torch.arange(width, device=device), indexing='ij')
                mask = ((y - cy)**2 + (x - cx)**2) < r**2
                u[b, 0, mask] = 0.5
                v[b, 0, mask] = 0.25
    
    # Ensure values are in valid range
    u = torch.clamp(u, 0, 1)
    v = torch.clamp(v, 0, 1)
    
    return u, v


class TuringPatternDataset(Dataset):
    """
    Gray-Scott Turing pattern dataset for diffusion models.
    
    Each sample contains both U and V concentration fields after evolving for a specified
    number of steps. Different parameter sets produce different pattern types.
    
    Returns 2-channel tensors by default (U and V), where:
    - Channel 0: U concentration (typically more uniform background)
    - Channel 1: V concentration (typically shows the patterns more clearly)
    
    The patterns are continuous-valued (not binary like Game of Life) and exhibit
    beautiful self-organizing structures like spots, stripes, spirals, and more.
    """
    
    PRESETS = {
        # Classical patterns from Pearson 1993 and other sources
        # Note: Du/Dv ratio is typically 2, as U diffuses faster than V
        'spots': {'Du': 0.16, 'Dv': 0.08, 'F': 0.035, 'k': 0.065},
        'dots': {'Du': 0.16, 'Dv': 0.08, 'F': 0.04, 'k': 0.07},
        'stripes': {'Du': 0.16, 'Dv': 0.08, 'F': 0.035, 'k': 0.060},
        'spirals': {'Du': 0.10, 'Dv': 0.05, 'F': 0.014, 'k': 0.054},
        'waves': {'Du': 0.16, 'Dv': 0.08, 'F': 0.014, 'k': 0.047}, # from https://arxiv.org/abs/1501.01990
        'holes': {'Du': 0.16, 'Dv': 0.08, 'F': 0.039, 'k': 0.058},
        'chaos': {'Du': 0.10, 'Dv': 0.05, 'F': 0.026, 'k': 0.051},
        'maze': {'Du': 0.16, 'Dv': 0.08, 'F': 0.029, 'k': 0.057},

        # Interesting patterns, often needs 128 x 128 resolution
        'life': {'Du': 0.16, 'Dv': 0.08, 'F': 0.006, 'k': 0.0450},
        
        # Additional patterns from various sources
        'bubbles': {'Du': 0.16, 'Dv': 0.08, 'F': 0.090, 'k': 0.059},
        'worms': {'Du': 0.16, 'Dv': 0.08, 'F': 0.046, 'k': 0.063},
        'solitons': {'Du': 0.10, 'Dv': 0.05, 'F': 0.030, 'k': 0.060},
        'pulsating_solitons': {'Du': 0.10, 'Dv': 0.05, 'F': 0.025, 'k': 0.060},
        'u_skate': {'Du': 0.16, 'Dv': 0.08, 'F': 0.062, 'k': 0.061},
        'fingerprints': {'Du': 0.16, 'Dv': 0.08, 'F': 0.040, 'k': 0.064},
    }
    
    def __init__(
        self,
        *,
        num_samples: int,
        height: int,
        width: int,
        steps: int = 1000,
        Du: float = 0.16,
        Dv: float = 0.08,
        F: float = 0.035,
        k: float = 0.065,
        dt: float = 1.0,
        clamp: bool = True,
        pattern_preset: str = None,
        init_pattern: str = 'multiple_seeds',
        device: str = "cpu",
        mode: str = "precompute",
        chunk: int = 32,
        normalize: bool = True,
        return_channel: str = 'both',  # 'u', 'v', or 'both'
        precomputed_data: torch.Tensor = None,
        mu: float = 0.0,
        std: float = 1.0
    ):
        """
        Args:
            num_samples: Number of patterns to generate
            height, width: Pattern dimensions
            steps: Number of simulation steps
            Du, Dv: Diffusion coefficients for U and V (default: 0.16, 0.08)
            F: Feed rate (default: 0.035)
            k: Kill rate (default: 0.065)
            dt: Time step (default: 1.0)
            pattern_preset: Use preset parameters including Du, Dv, F, k ('spots', 'stripes', etc.)
            init_pattern: Initial condition type
            device: Computation device
            mode: 'precompute' or 'online'
            chunk: Batch size for precomputation
            normalize: Whether to normalize the data
            return_channel: Which concentration field to return (default: 'both' for 2 channels)
            precomputed_data: Pre-computed data tensor to use instead of running simulation
            mu, std: Normalization parameters when using precomputed_data
        """
        super().__init__()
        self.num_samples = num_samples
        self.height, self.width = height, width
        self.steps = steps
        self.init_pattern = init_pattern
        self.device = device
        self.mode = mode
        self.return_channel = return_channel
        self.mu = mu
        self.std = std
        
        # Use preset parameters if specified
        if pattern_preset and pattern_preset in self.PRESETS:
            preset = self.PRESETS[pattern_preset]
            Du = preset['Du']
            Dv = preset['Dv']
            F = preset['F']
            k = preset['k']
        
        # Create simulator
        self.sim = GrayScottSimulator(Du=Du, Dv=Dv, F=F, k=k, dt=dt, clamp=clamp, device=device)
        
        # Store parameters
        self.params = {
            'Du': Du, 'Dv': Dv, 'F': F, 'k': k, 'dt': dt,
            'pattern_preset': pattern_preset
        }
        
        if precomputed_data is not None:
            # Use provided precomputed data
            self.data = precomputed_data
        elif mode == "precompute":
            # Generate data through simulation
            self.data = self._precompute(chunk)
            if normalize:
                self.data = self.normalize(self.data)
        elif mode != "online":
            raise ValueError("mode must be 'precompute' or 'online'")
    
    def save(self, filepath):
        """
        Save the complete dataset including all parameters and precomputed data.
        
        Args:
            filepath: Path to save the dataset (will save as .pt file)
        """
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not filepath.endswith('.pt'):
            filepath += '.pt'
            
        # Prepare data to save
        save_data = {
            'num_samples': self.num_samples,
            'height': self.height,
            'width': self.width,
            'steps': self.steps,
            'init_pattern': self.init_pattern,
            'device': self.device,
            'mode': self.mode,
            'return_channel': self.return_channel,
            'params': self.params,
            'mu': getattr(self, 'mu', 0.0),
            'std': getattr(self, 'std', 1.0),
        }
        
        # Include precomputed data if available
        if hasattr(self, 'data'):
            save_data['data'] = self.data
        
        torch.save(save_data, filepath)
        print(f"Dataset saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a previously saved dataset. No simulation will be performed during loading.
        
        Args:
            filepath: Path to the saved dataset file
            
        Returns:
            TuringPatternDataset instance with all data and parameters restored
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        # Load the saved data
        save_data = torch.load(filepath, map_location='cpu')
        
        # Extract parameters
        params = save_data['params']
        
        # Create instance using precomputed_data parameter
        instance = cls(
            num_samples=save_data['num_samples'],
            height=save_data['height'],
            width=save_data['width'],
            steps=save_data['steps'],
            Du=params['Du'],
            Dv=params['Dv'],
            F=params['F'],
            k=params['k'],
            dt=params['dt'],
            pattern_preset=params['pattern_preset'],
            init_pattern=save_data['init_pattern'],
            device=save_data['device'],
            mode=save_data['mode'],
            return_channel=save_data['return_channel'],
            precomputed_data=save_data.get('data', None),
            mu=save_data['mu'],
            std=save_data['std']
        )
        
        print(f"Dataset loaded from {filepath}")
        print(f"Pattern type: {params.get('pattern_preset', 'custom')}")
        print(f"Parameters: Du={params['Du']}, Dv={params['Dv']}, F={params['F']}, k={params['k']}")
        print(f"Samples: {instance.num_samples}, Size: {instance.height}x{instance.width}")
        
        return instance
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.mode == "precompute":
            return self.data[idx]
        else:  # online
            return self._simulate(1).squeeze(0)
    
    def normalize(self, x):
        """Normalize data to zero mean and unit variance."""
        mu = x.transpose(0, 1).reshape(2, -1).mean(dim=-1).reshape(1, 2, 1, 1)
        std = x.transpose(0, 1).reshape(2, -1).std(dim=-1).reshape(1, 2, 1, 1)
        self.mu, self.std = mu, std
        return (x - mu) / (std + 1e-5)
    
    def denormalize(self, x):
        """Reverse normalization."""
        return x * (self.std + 1e-5) + self.mu
    
    @torch.no_grad()
    def _simulate(self, batch_size: int) -> torch.Tensor:
        """Generate patterns through simulation."""
        # Create initial conditions
        u0, v0 = create_random_state(
            batch_size, self.height, self.width,
            pattern_type=self.init_pattern, device=self.device
        )
        
        # Evolve the system
        u_states, v_states = self.sim.forward(u0, v0, steps=self.steps)
        
        # Extract final state
        if self.return_channel == 'u':
            return u_states[:, -1]  # shape (B, 1, H, W)
        elif self.return_channel == 'v':
            return v_states[:, -1]  # shape (B, 1, H, W)
        else:  # 'both'
            return torch.cat([u_states[:, -1], v_states[:, -1]], dim=1)  # shape (B, 2, H, W)
    
    def _precompute(self, chunk: int) -> torch.Tensor:
        """Generate samples in chunks to manage memory."""
        
        samples = []
        remaining = self.num_samples
        
        # Create progress bar
        pbar = tqdm(total=self.num_samples, desc="Generating samples")
        
        while remaining > 0:
            cur = min(chunk, remaining)
            batch = self._simulate(cur).cpu()  # Move to CPU to save GPU memory
            samples.append(batch)
            remaining -= cur
            
            # Update progress bar
            pbar.update(cur)
        
        pbar.close()
        return torch.cat(samples, dim=0).cpu()
    
    def get_params(self):
        """Return the Gray-Scott parameters used."""
        return self.params.copy()