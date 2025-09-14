import torch
from torch.utils.data import Dataset, DataLoader
from utils import integrate_rk4


def vf_lorenz(sigma, rho, beta):
    def vf(state):
        x, y, z = state[0], state[1], state[2]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return torch.tensor([dx, dy, dz])

    return vf


def generate_lorenz_time_series(
    num_steps=10000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0, initial_state=None
):
    """
    Generate time series data from the Lorenz system using PyTorch.

    Args:
        num_steps (int): Number of time steps to simulate
        dt (float): Time step size
        sigma (float): Sigma parameter of the Lorenz system
        rho (float): Rho parameter of the Lorenz system
        beta (float): Beta parameter of the Lorenz system
        initial_state (torch.Tensor, optional): Initial state [x, y, z].
                                               If None, random initialization is used.

    Returns:
        torch.Tensor: Time series data with shape [num_steps, 3]
    """
    # Initialize state
    if initial_state is None:
        state = torch.randn(3)  # Random initial state
    else:
        state = initial_state.clone()

    # Prepare tensor for storing the time series
    time_series = torch.zeros((num_steps, 3))
    time_series[0] = state

    # Generate the time series
    f = vf_lorenz(sigma, rho, beta)
    for i in range(1, num_steps):
        state = integrate_rk4(f, state, dt)
        time_series[i] = state

    return time_series


class LorenzDataset(Dataset):
    """
    Dataset for accessing individual points in a Lorenz attractor time series.
    """

    def __init__(
        self,
        num_simulations=10,
        num_steps=5000,
        dt=0.01,
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
    ):
        """
        Initialize the dataset by generating a single Lorenz time series.

        Args:
            num_steps: Number of time steps in the series
            dt: Time step for integration
            sigma, rho, beta: Lorenz system parameters
        """
        self.time_series = []
        for i in range(num_simulations):
            self.time_series.append(
                generate_lorenz_time_series(
                    num_steps=num_steps, dt=dt, sigma=sigma, rho=rho, beta=beta
                )
            )
        self.time_series = torch.cat(self.time_series, dim=0)
        self.time_series = self.normalize(self.time_series)
        self.dim = 3

    def normalize(self, time_series):
        """Normalize the time series to have zero mean and unit variance."""
        # if mean and std are not exist
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            self.mean = time_series.mean(dim=0)
            self.std = time_series.std(dim=0)
        return (time_series - self.mean) / self.std

    def denormalize(self, normalized_series):
        """Denormalize the time series to recover the original values."""
        return normalized_series * self.std + self.mean

    def __len__(self):
        """Return the number of points in the time series."""
        return len(self.time_series)

    def __getitem__(self, idx):
        """
        Get a 3D data point at the specified time step.

        Args:
            idx: Index of the time step

        Returns:
            3D data point [x, y, z] at time step idx
        """
        return self.time_series[idx]


class RingDataset(Dataset):
    """
    Dataset for generating points uniformly distributed in a 2D ring.
    """

    def __init__(self, num_samples=10000, r_min=1.0, r_max=2.0):
        """
        Initialize the dataset by generating points in a ring.

        Args:
            num_samples: Number of points to generate
            r_min: Inner radius of the ring
            r_max: Outer radius of the ring
        """
        self.num_samples = num_samples
        self.r_min = r_min
        self.r_max = r_max
        self.time_series = self._generate_ring_points()
        self.dim = 2

    def _generate_ring_points(self):
        """
        Generate points uniformly distributed in a ring.

        Returns:
            Tensor of shape [num_samples, 2] containing points in the ring
        """
        # Generate random angles
        theta = 2 * torch.pi * torch.rand(self.num_samples)

        # Generate random radii with correct distribution for uniform density
        # For uniform distribution in a ring, we need to sample r^2 uniformly
        r_squared = (
            torch.rand(self.num_samples) * (self.r_max**2 - self.r_min**2)
            + self.r_min**2
        )
        r = torch.sqrt(r_squared)

        # Convert to Cartesian coordinates
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        return torch.stack([x, y], dim=1)

    def __len__(self):
        """Return the number of points in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a 2D data point at the specified index.

        Args:
            idx: Index of the point

        Returns:
            2D data point [x, y] at index idx
        """
        return self.time_series[idx]


class TwoPeaksDataset(Dataset):
    """
    Dataset for generating points in a 2D two-peaks distribution.
    """

    def __init__(self, num_samples=10000, centers=None, std=0.5):
        """
        Initialize the TwoPeaksDataset.

        Args:
            num_samples: Number of points to generate
            centers: Centers of the two Gaussian distributions, defaults to [(-1, -1), (1, 1)]
            std: Standard deviation of the Gaussian distributions
        """
        self.num_samples = num_samples
        self.centers = centers if centers is not None else [(-1, -1), (1, 1)]
        self.std = std
        self.time_series = self._generate_two_peaks_points()
        self.dim = 2

    def _generate_two_peaks_points(self):
        """
        Generate points from a mixture of two Gaussian distributions.

        Returns:
            Tensor of shape [num_samples, 2] containing points from the distribution
        """
        # Determine how many samples to generate from each Gaussian
        samples_per_peak = self.num_samples // 2

        # Generate samples for the first Gaussian
        points_1 = self._generate_gaussian_points(self.centers[0], samples_per_peak)

        # Generate samples for the second Gaussian
        points_2 = self._generate_gaussian_points(
            self.centers[1], self.num_samples - samples_per_peak
        )

        # Combine the samples
        return torch.cat([points_1, points_2], dim=0)

    def _generate_gaussian_points(self, center, num_points):
        """
        Generate points from a single Gaussian distribution.

        Args:
            center: Center of the Gaussian distribution (x, y)
            num_points: Number of points to generate

        Returns:
            Tensor of shape [num_points, 2] containing points from the Gaussian
        """
        # Generate random points from a standard normal distribution
        points = torch.randn(num_points, 2) * self.std

        # Shift the points to the specified center
        points[:, 0] += center[0]
        points[:, 1] += center[1]

        return points

    def __len__(self):
        """Return the number of points in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a 2D data point at the specified index.

        Args:
            idx: Index of the point

        Returns:
            2D data point [x, y] at index idx
        """
        return self.time_series[idx]


class TwoMoonsDataset(Dataset):
    """
    Dataset that generates samples from a two moons distribution.

    This dataset creates points in a 2D space arranged in two half-moon shapes.
    """

    def __init__(self, num_samples=5000, noise=0.1, vertical_gap=0.0):
        """
        Initialize the TwoMoonsDataset.

        Args:
            num_samples (int): Number of data points to generate
            noise (float): Standard deviation of Gaussian noise added to the points
            vertical_gap (float): Vertical distance between the two moons (smaller values bring moons closer)
        """
        super(TwoMoonsDataset, self).__init__()
        self.num_samples = num_samples
        self.noise = noise
        self.vertical_gap = vertical_gap
        self.dim = 2
        self.time_series = self._generate_two_moons()
        self.time_series = self.normalize(self.time_series)

    def normalize(self, time_series):
        """Normalize the time series to have zero mean and unit variance."""
        mean = time_series.mean(dim=0)
        std = time_series.std(dim=0)
        return (time_series - mean) / std

    def _generate_two_moons(self):
        """
        Generate points arranged in two half-moon shapes.

        Returns:
            torch.Tensor: Tensor of shape [num_samples, 2] containing the generated points
        """
        # Determine how many samples to generate for each moon
        samples_per_moon = self.num_samples // 2

        # Generate the first moon
        points_1 = self._generate_moon_points(0, samples_per_moon)

        # Generate the second moon
        points_2 = self._generate_moon_points(1, self.num_samples - samples_per_moon)

        # Combine the samples
        return torch.cat([points_1, points_2], dim=0)

    def _generate_moon_points(self, moon_idx, num_points):
        """
        Generate points for one half-moon shape.

        Args:
            moon_idx (int): Index of the moon (0 or 1)
            num_points (int): Number of points to generate

        Returns:
            torch.Tensor: Tensor of shape [num_points, 2] containing points for one moon
        """
        # Generate angles from 0 to pi
        angles = torch.linspace(0, torch.pi, num_points)

        # Create the moon shape using trigonometric functions
        x = torch.cos(angles)
        y = torch.sin(angles)

        # Create the points tensor
        points = torch.stack([x, y], dim=1)

        # Apply transformations to create the two moons shape with opposite orientations
        half_gap = self.vertical_gap / 2
        if moon_idx == 1:
            # Second moon: flip horizontally and vertically, shift right and down
            points[:, 0] = 1.0 - points[:, 0]  # Flip horizontally and shift right
            points[:, 1] = -points[:, 1] - half_gap  # Flip vertically and shift down
        else:
            # First moon: just shift up
            points[:, 1] = points[:, 1] + half_gap  # Shift up

        # Add Gaussian noise
        points = points + torch.randn_like(points) * self.noise

        return points

    def __len__(self):
        """Return the number of points in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a 2D data point at the specified index.

        Args:
            idx: Index of the point

        Returns:
            2D data point [x, y] at index idx
        """
        return self.time_series[idx]
