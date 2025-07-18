import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def _pairwise_diffs(x):
    """Return pairwise vector differences and squared distances.

    Parameters
    ----------
    x : Tensor (B, N, 2) of positions

    Returns
    -------
    diff : (B, N, N, 2)
    dist2 : (B, N, N)
    """
    diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,2)  p_j - p_i
    dist2 = (diff**2).sum(-1)              # (B,N,N)
    return diff, dist2


class Boids(nn.Module):
    """Vectorised 2‑D Boids simulator (Reynolds, 1987).

    Each boid obeys three rules:
    1. **Alignment** – steer toward the average heading of neighbours.
    2. **Cohesion**  – steer toward the centre of mass of neighbours.
    3. **Separation** – avoid crowding neighbours too closely.

    Parameters
    ----------
    num_boids : int
    radius : float, optional (default = 0.1)
        Interaction / perception radius (in world units; world is a unit square).
    align_w, coh_w, sep_w : float, optional
        Weights for alignment, cohesion, and separation steering components.
    max_speed : float, optional (default = 0.03)
        Maximum speed magnitude (unit square per time‑step).
    max_force : float, optional (default = 0.05)
        Maximum steering acceleration magnitude.
    dt : float, optional (default = 1.0)
        Time‑step for Euler integration.
    boundary : {'torus', 'reflect'}, optional
        Boundary condition. "torus" wraps around; "reflect" bounces.
    device : str, optional
    """

    def __init__(
        self,
        num_boids: int,
        *,
        radius: float = 0.1,
        align_w: float = 1.0,
        coh_w: float = 1.0,
        sep_w: float = 1.5,
        max_speed: float = 0.3,
        max_force: float = 0.5,
        noise: float = 0.0,
        dt: float = 1.0,
        boundary: str = "torus",
        device: str = "cpu",
    ):
        super().__init__()
        self.n = num_boids
        self.r2 = radius ** 2
        self.aw, self.cw, self.sw = align_w, coh_w, sep_w
        self.max_speed = max_speed
        self.max_force = max_force
        self.noise = noise
        self.dt = dt
        if boundary not in {"torus", "reflect"}:
            raise ValueError("boundary must be 'torus' or 'reflect'")
        self.boundary = boundary
        self.device = device
        self.to(device)

    @torch.no_grad()
    def _step(self, pos: torch.Tensor, vel: torch.Tensor):
        """One Euler step.

        Parameters
        ----------
        pos : (B, N, 2)
        vel : (B, N, 2)
        Returns
        -------
        new_pos, new_vel : tuple of tensors with same shape.
        """
        B, N, _ = pos.shape
        diff, dist2 = _pairwise_diffs(pos)      # (B,N,N,2), (B,N,N)
        mask = (dist2 < self.r2) & (dist2 > 0)   # exclude self (dist2>0)
        mask_f = mask.float()
        # --- alignment -----------------------------------------------------
        neigh_vel_sum = (vel.unsqueeze(1) * mask_f.unsqueeze(-1)).sum(2)
        neigh_cnt = mask_f.sum(2, keepdim=True)
        avg_vel = neigh_vel_sum / (neigh_cnt + 1e-6)
        align_force = avg_vel - vel
        # --- cohesion ------------------------------------------------------
        neigh_pos_sum = (pos.unsqueeze(1) * mask_f.unsqueeze(-1)).sum(2)
        coh_center = neigh_pos_sum / (neigh_cnt + 1e-6)
        coh_force = coh_center - pos
        # --- separation ----------------------------------------------------
        # Normalised repulsion vector inversely proportional to distance
        repulsion = (diff / (dist2.unsqueeze(-1) + 1e-6)) * mask_f.unsqueeze(-1)
        sep_force = -repulsion.sum(2)
        # --- noise on direction --------------------------------------------
        if self.noise > 0.0:
            noise = torch.randn_like(vel) * self.noise
            vel = vel + noise * self.dt
            # re-normalise velocity to max speed
            speed = torch.norm(vel, dim=-1, keepdim=True)
            vel = torch.where(speed > self.max_speed,
                              vel / speed * self.max_speed,
                              vel)
        # --- combine -------------------------------------------------------
        steer = (
            self.aw * align_force +
            self.cw * coh_force +
            self.sw * sep_force
        )
        # limit steering magnitude
        steer_norm = torch.norm(steer, dim=-1, keepdim=True)
        steer = torch.where(steer_norm > self.max_force,
                            steer / steer_norm * self.max_force,
                            steer)
        # update velocity
        vel_new = vel + steer * self.dt
        # limit speed
        speed = torch.norm(vel_new, dim=-1, keepdim=True)
        vel_new = torch.where(speed > self.max_speed,
                              vel_new / speed * self.max_speed,
                              vel_new)
        # update position
        pos_new = pos + vel_new * self.dt
        if self.boundary == "torus":
            pos_new = pos_new % 1.0
        else:  # reflect
            over = (pos_new > 1.0) | (pos_new < 0.0)
            pos_new = torch.where(over, torch.clamp(pos_new, 0.0, 1.0), pos_new)
            vel_new = torch.where(over, -vel_new, vel_new)
        return pos_new, vel_new

    def forward(self, pos: torch.Tensor, vel: torch.Tensor, steps: int = 1, trajectory: bool = False):
        """Evolve the flock for `steps` time‑steps.

        Parameters
        ----------
        pos, vel : tensors of shape (B, N, 2)
        steps : int
        Returns
        -------
        new_pos, new_vel : tensors of shape (B, N, 2)
        """
        cur_pos, cur_vel = pos, vel
        if trajectory:
            all_pos, all_vel = [cur_pos], [cur_vel]
            for _ in range(steps):
                cur_pos, cur_vel = self._step(cur_pos, cur_vel)
                all_pos.append(cur_pos)
                all_vel.append(cur_vel)
            return torch.stack(all_pos, dim=1), torch.stack(all_vel, dim=1)
        # no trajectory, just final state
        for _ in range(steps):
            cur_pos, cur_vel = self._step(cur_pos, cur_vel)
        return cur_pos, cur_vel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_random_state(num_boids: int, *, device="cpu", max_speed=0.03):
    """Uniform random positions in unit square and random velocity directions.

    Returns
    -------
    pos, vel : tensors of shape (1, N, 2) suitable for Boids.forward
    """
    pos = torch.rand(1, num_boids, 2, device=device)
    angles = 2 * torch.pi * torch.rand(1, num_boids, 1, device=device)
    vel = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1) * max_speed
    return pos, vel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BoidsDataset(Dataset):
    """Boids snapshots for generative modelling / score matching.

    Each sample := (pos, vel) where both tensors have shape (num_boids, 2).
    Only the converged state (after `steps` iterations) is stored / returned.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        num_boids: int,
        steps: int = 100,
        device: str = "cpu",
        mode: str = "precompute",       # "precompute" | "online"
        chunk: int = 32,
        normalize: bool = False,
        boids_params: dict | None = None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.steps = steps
        self.num_boids = num_boids
        self.device = device
        self.mode = mode

        boids_kwargs = boids_params or {}
        self.sim = Boids(num_boids, device=device, **boids_kwargs)

        if mode == "precompute":
            pos, vel = self._precompute(chunk)
            if normalize:
                self.pos_data, self.vel_data = self._normalize(pos), self._normalize(vel)
            else:
                self.pos_data, self.vel_data = pos, vel
            # store stats for denormalisation even if not used
            self.mu = torch.tensor(0.0); self.std = torch.tensor(1.0)
        elif mode != "online":
            raise ValueError("mode must be 'precompute' or 'online'")

    # ------------- dataset api --------------------------------------------
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == "precompute":
            return self.pos_data[idx], self.vel_data[idx]
        else:  # online
            pos, vel = self._simulate(1)
            return pos[0], vel[0]

    # ------------- helpers -------------------------------------------------
    def _normalize(self, x):
        mu = x.mean()
        std = x.std()
        self.mu, self.std = mu, std
        return (x - mu) / (std + 1e-8)

    def denormalize(self, x):
        return x * (self.std + 1e-8) + self.mu

    @torch.no_grad()
    def _simulate(self, batch_size: int):
        pos0, vel0 = create_random_state(self.num_boids, device=self.device,
                                         max_speed=self.sim.max_speed)
        if batch_size > 1:
            pos0 = pos0.repeat(batch_size, 1, 1)
            vel0 = vel0.repeat(batch_size, 1, 1)
        posT, velT = self.sim.forward(pos0, vel0, steps=self.steps)
        # shapes: (B, N, 2)
        return posT.cpu(), velT.cpu()

    def _precompute(self, chunk: int):
        pos_samples = []
        vel_samples = []
        remaining = self.num_samples
        while remaining:
            cur = min(chunk, remaining)
            pos, vel = self._simulate(cur)
            pos_samples.append(pos)
            vel_samples.append(vel)
            remaining -= cur
        return torch.cat(pos_samples, dim=0), torch.cat(vel_samples, dim=0)
