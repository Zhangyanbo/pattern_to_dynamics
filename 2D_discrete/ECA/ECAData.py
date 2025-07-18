import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ElementaryCA(nn.Module):
    """
    Fast Elementary Cellular Automaton simulator using pure PyTorch.

    State is 1‑D: tensor of shape (batch_size, 1, width) with 0/1 values.
    Rule is an integer 0–255. Neighborhood mapping follows Wolfram numbering:
    bit 7 -> 111, …, bit 0 -> 000.
    padding_mode: 'zeros' for fixed dead boundary, 'circular' for wrap‑around.
    """

    def __init__(self, rule: int = 110, device: str = "cpu", padding_mode: str = "zeros"):
        super().__init__()
        if not (0 <= rule <= 255):
            raise ValueError("rule must be between 0 and 255.")
        self.rule = rule
        self.device = device
        assert padding_mode in ["zeros", "circular"], \
            "padding_mode must be 'zeros' or 'circular'."
        self.padding_mode = padding_mode

        # Pre‑compute rule table as buffer (size 8)
        rule_bits = torch.tensor([(rule >> i) & 1 for i in range(8)],
                                 dtype=torch.float32, device=device)  # index 0..7
        # rule_bits[i] gives output for neighborhood with value i
        self.register_buffer("rule_table", rule_bits)

        self.to(device)

    def _next_state(self, state: torch.Tensor) -> torch.Tensor:
        """Compute next ECA state.

        Args:
            state: Tensor (B, 1, W) with 0/1 values

        Returns:
            Tensor (B, 1, W) – next state
        """
        pad = 1
        if self.padding_mode == "circular":
            padded = F.pad(state, (pad, pad), mode="circular")
        elif self.padding_mode == "zeros":
            padded = F.pad(state, (pad, pad), mode="constant", value=0)
        else:
            raise ValueError("padding_mode must be 'zeros' or 'circular'")

        left = padded[..., :-2]
        center = padded[..., 1:-1]
        right = padded[..., 2:]

        idx = (left * 4 + center * 2 + right).long()  # value in 0‑7
        next_state = self.rule_table[idx]
        return next_state

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return self._next_state(state)

    def forward(self, state: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Evolve `state` for `steps` time‑steps.

        Returns
        -------
        Tensor of shape (B, steps+1, 1, W) – includes initial state.
        """
        states = [state]
        cur = state
        for _ in range(steps):
            cur = self.step(cur)
            states.append(cur)
        return torch.stack(states, dim=1)


def create_random_state(batch_size: int, width: int, density: float = 0.5, device: str = "cpu"):
    """Random binary 1‑D state tensor (B, 1, W) with given alive‑cell density."""
    rnd = torch.rand(batch_size, 1, width, device=device)
    return (rnd < density).float()


class ECADataset(Dataset):
    """
    Elementary Cellular Automaton snapshots for diffusion / score‑matching.

    Each sample: tensor[1, W] representing the final state after `steps` evolution.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        width: int,
        steps: int = 1,
        density: float = 0.5,
        rule: int = 110,
        device: str = "cpu",
        mode: str = "precompute",          # "precompute" | "online"
        chunk: int = 2048,                 # batch size for precompute
        normalize: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.num_samples, self.steps = num_samples, steps
        self.width, self.density = width, density
        self.device, self.mode = device, mode

        # Reuse the same simulator instance
        self.sim = ElementaryCA(rule=rule, device=device, padding_mode=padding_mode)

        if mode == "precompute":
            raw = self._precompute(chunk)
            if normalize:
                self.data = self._normalize(raw)
            else:
                self.data = raw
                self.mu = 0.0
                self.std = 1.0
        elif mode != "online":
            raise ValueError("mode must be 'precompute' or 'online'")

    # ---------- dataset api ----------
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == "precompute":
            return self.data[idx]
        else:
            return self._simulate(1).squeeze(0)

    # ---------- helpers ----------
    def _normalize(self, x):
        mu = x.mean()
        std = x.std()
        self.mu, self.std = mu, std
        return (x - mu) / std

    def denormalize(self, x):
        return x * self.std + self.mu

    @torch.no_grad()
    def _simulate(self, batch_size: int) -> torch.Tensor:
        x0 = create_random_state(batch_size, self.width,
                                 density=self.density, device=self.device)
        xT = self.sim.forward(x0, steps=self.steps)[:, -1]  # final state only
        return xT  # (B,1,W)

    def _precompute(self, chunk: int) -> torch.Tensor:
        samples = []
        remaining = self.num_samples
        while remaining:
            cur = min(chunk, remaining)
            samples.append(self._simulate(cur).cpu())
            remaining -= cur
        return torch.cat(samples, dim=0)
