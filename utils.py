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


def integrate_rk4(vf, pos, dt):
    """
    Performs one step of the 4th-order Runge-Kutta integration method.
    
    Args:
        vf: A function that takes a position tensor and returns a velocity vector.
        pos: The current position tensor.
        dt: The time step for integration (dt).
        
    Returns:
        The new position after one integration step.
    """
    k1 = vf(pos)
    k2 = vf(pos + k1 * dt / 2)
    k3 = vf(pos + k2 * dt / 2)
    k4 = vf(pos + k3 * dt)
    return pos + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


def _hutchinson_estimate(vjp_func, x, z):
    vjp = vjp_func(z)[0]
    return torch.einsum('nd, nd -> n', vjp, z)

def hutchinson_estimate(model, x, num_samples=2):
    output, vjp_func= torch.func.vjp(model, x)
    z = torch.randn((num_samples, *x.shape), device=x.device)
    est = torch.vmap(_hutchinson_estimate, (None, None, 0))(vjp_func, x, z)
    return est.mean(dim=0), output

def div_estimate(flow_model, score_model, x, t, num_samples=2):
    num_batch = x.shape[0]
    t = torch.ones((num_batch,), device=x.device) * t
    xt, eps = score_model.diffuse(x, t)
    with torch.no_grad():
        s = score_model.score(xt, t)
    div_term, v = hutchinson_estimate(flow_model, xt, num_samples=num_samples)
    oth_term = torch.einsum('nd, nd -> n', v, s)
    return div_term + oth_term, s, v, div_term, oth_term