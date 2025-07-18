import torch
import torch.nn as nn

class VPJBatchNorm(nn.BatchNorm1d):
    def __init__(self, *args, use_mean=True, use_var=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mean = use_mean
        self.use_var = use_var

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Determine if we're in training mode and tracking running stats.
        if self.training and self.track_running_stats:
            # Compute batch statistics.
            # For 2D input: (N, C), for 3D input: (N, C, L)
            if input.dim() == 2:
                batch_mean = input.mean(0)
                batch_var = input.var(0, unbiased=False)
            elif input.dim() == 3:
                batch_mean = input.mean([0, 2])
                batch_var = input.var([0, 2], unbiased=False)
            else:
                raise ValueError("VPJBatchNorm expects input of dimension 2 or 3.")

            # Update running statistics without in-place mutation.
            with torch.no_grad():
                self.num_batches_tracked = self.num_batches_tracked + 1
                exponential_average_factor = self.momentum

                new_running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * batch_mean
                new_running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * batch_var

                self.running_mean = new_running_mean.detach()
                self.running_var = new_running_var.detach()

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Reshape mean and var to allow broadcasting over input.
        if input.dim() == 2:
            mean = mean.view(1, -1)
            var = var.view(1, -1)
        elif input.dim() == 3:
            mean = mean.view(1, -1, 1)
            var = var.view(1, -1, 1)
        
        # if use_mean is False, set mean to 0
        if not self.use_mean:
            mean = torch.zeros_like(mean)
        # if use_var is False, set var to 1
        if not self.use_var:
            var = torch.ones_like(var)

        # Normalize the input.
        normalized = (input - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            # Reshape weight and bias for broadcasting.
            if input.dim() == 2:
                normalized = normalized * self.weight.view(1, -1) + self.bias.view(1, -1)
            elif input.dim() == 3:
                normalized = normalized * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

        return normalized