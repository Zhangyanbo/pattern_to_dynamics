import torch
import torch.nn as nn

class VPJBatchNorm(nn.BatchNorm1d):
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

        # Normalize the input.
        normalized = (input - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            # Reshape weight and bias for broadcasting.
            if input.dim() == 2:
                normalized = normalized * self.weight.view(1, -1) + self.bias.view(1, -1)
            elif input.dim() == 3:
                normalized = normalized * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

        return normalized


class VPJBatchNorm2d(nn.BatchNorm2d):
    r"""
    Same functionality as nn.BatchNorm2d, but forward is rewritten to:
    1) Avoid in-place modifications to running_mean / running_var,
    2) Explicitly detach the computation graph from buffers - facilitates Jacobian-vector product.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 4:                       # (N, C, H, W)
            raise ValueError(
                f"VPJBatchNorm2d expects 4‑D input, got {input.dim()}‑D")

        # ----------------------------------------------
        # 1. Training mode: compute current batch statistics and update buffers
        # ----------------------------------------------
        if self.training and self.track_running_stats:
            # (N, C, H, W)  →  Average over N, H, W
            batch_mean = input.mean([0, 2, 3])                     # [C]
            batch_var  = input.var ([0, 2, 3], unbiased=False)     # [C]

            with torch.no_grad():
                self.num_batches_tracked = self.num_batches_tracked + 1

                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

                new_running_mean = (
                    1 - exponential_average_factor) * self.running_mean \
                    + exponential_average_factor * batch_mean
                new_running_var = (
                    1 - exponential_average_factor) * self.running_var \
                    + exponential_average_factor * batch_var

                # **Do NOT** change these two lines to in-place operations (+= or *=),
                # otherwise Autograd will issue warnings or even errors during JVP/VJP
                self.running_mean = new_running_mean.detach()
                self.running_var  = new_running_var.detach()

            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

        # ----------------------------------------------
        # 2. Normalize + (optional) Affine
        # ----------------------------------------------
        mean = mean.view(1, -1, 1, 1)     # [1, C, 1, 1] 便于广播
        var  = var.view (1, -1, 1, 1)

        normalized = (input - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            normalized = normalized * self.weight.view(1, -1, 1, 1) \
                                       + self.bias.view  (1, -1, 1, 1)
        return normalized
