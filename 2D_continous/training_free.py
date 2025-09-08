import torch
import torch.nn as nn
import torch.nn.functional as F
from train_dynamics import load_score_model
from turing_pattern import TuringPatternDataset
from tqdm import tqdm


def score_function(unet, scheduler, t):
    def score_model(xt, return_sample=False):
        model_output = unet(xt, t)
        x0 = scheduler.step(model_output.sample, t, xt).pred_original_sample
        alpha_t = scheduler.alphas_cumprod[t]
        score = (x0 * alpha_t ** 0.5 - xt) / (1 - alpha_t)
        if return_sample:
            return score, model_output.sample
        else:
            return score

    return score_model

def load_models(name: str, use_bn: bool = False):
    dataset = TuringPatternDataset.load(f'./turing_pattern/data/{name}_128x128.pt')
    score_model, scheduler = load_score_model(name, device='cuda')

    return dataset, score_model, scheduler

def training_free_flow(name:str, k=None, num_channel:int=2, t:int=10) -> callable:
    dataset, score_model, scheduler = load_models(name)
    if k is None:
        p = torch.randn(num_channel, num_channel)
        k = p - p.t()
        k = k.reshape(num_channel, num_channel, 1, 1)
        k = k / k.norm()
    
    sf = score_function(score_model, scheduler, t=t)
    
    def flow_model(x):
        s = sf(x)
        v = F.conv2d(s, k.to(s.device), padding=0)
        return v
    return flow_model, score_model, dataset, scheduler

def sde_solve(v:callable, s:callable, scheduler, x, T:int, dt:float=0.1, eta:float=0.1, v_factor:float=1.0, tau:int=10, denoise:bool=True) -> torch.Tensor:
    """
    Solve the SDE: dx = (v(x) + eta * s(x)) dt + sqrt(2 * eta * dt) dW
    using Huen's method.

    Args:
        v: The flow model function.
        s: The score model function.
        x: The current position tensor.
        dt: Time step for the integration.
        eta: Weight for the score term.

    Returns:
        The new position after one integration step.
    """
    sf = score_function(s, scheduler, tau)
    def a(x):
        score, sample = sf(x, return_sample=True)
        return v(x) + eta * score, sample

    trace = []
    if denoise:
        trace_pred = []
    
    for t in tqdm(range(T)):
        eps_t = torch.randn_like(x)
        a_t, sample = a(x)
        trace.append(x.cpu())
        if denoise:
            x_pred = scheduler.step(sample, tau, x).pred_original_sample
            if t != 0:
                trace_pred.append(x_pred.cpu().clamp(-1, 1))
            else:
                trace_pred.append(x.cpu().clamp(-1, 1))
        noise_level = (2 * eta * dt) ** 0.5
        x_hat = x + a_t * dt + noise_level * eps_t
        x = x + (a_t + a(x_hat)[0]) / 2 * dt + noise_level * eps_t
    
    if denoise:
        trace_pred = torch.stack(trace_pred, dim=0)
    
    if denoise:
        return torch.stack(trace, dim=0), trace_pred
    else:
        return torch.stack(trace, dim=0)