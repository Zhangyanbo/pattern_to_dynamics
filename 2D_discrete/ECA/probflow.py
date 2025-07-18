import torch
import torch.nn as nn


class DivOutput:
    def __init__(self, vjp_term, oth_term, score, flow):
        self.vjp_term = vjp_term
        self.oth_term = oth_term
        self.score = score
        self.flow = flow

        self.div = self.vjp_term + self.oth_term
        self.loss = torch.mean(self.div ** 2) / flow.shape[-1]

        self.flow_energy = torch.mean(self.flow.pow(2))
    
    def __repr__(self):
        return f"DivOutput(loss={self.loss}, div={self.div}, score={self.score}, flow={self.flow})"

def _hutchinson_estimate(vjp_func, x, z):
    vjp = vjp_func(z)[0]
    return torch.einsum('nd, nd -> n', vjp, z)

def hutchinson_estimate(model, x, num_samples=2):
    output, vjp_func= torch.func.vjp(model, x)
    z = torch.randn((num_samples, *x.shape), device=x.device)
    est = torch.vmap(_hutchinson_estimate, (None, None, 0))(vjp_func, x, z)
    return est.mean(dim=0), output

def div_estimate(flow_model, score_model, x, num_samples=2):
    """
    Estimates the divergence of a flow model using the Hutchinson trace estimator.
    
    Args:
        flow_model: A neural network R^n -> R^n that outputs velocity vectors (with batch dim).
        score_model: A neural network R^n -> R^n that outputs score vectors (with batch dim).
        x: Input tensor of shape (batch_size, ...).
        num_samples: Number of random samples for Hutchinson estimator.
        
    Returns:
        Tuple containing:
        - Total divergence estimate (div_term + oth_term)
        - Score vectors from score_model
        - Velocity vectors from flow_model
        - Divergence term from Hutchinson estimator
        - Other term from score model interaction
    """
    with torch.no_grad():
        score = score_model(x)
    vjp_term, flow = hutchinson_estimate(flow_model, x, num_samples=num_samples)
    oth_term = torch.einsum('nd, nd -> n', flow, score)
    return DivOutput(vjp_term, oth_term, score, flow)