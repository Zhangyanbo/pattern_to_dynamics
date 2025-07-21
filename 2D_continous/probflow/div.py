import torch
from torch import vmap
from torch.func import jvp, vmap

def hutchinson_estimate(model, x, num_samples: int = 64):
    z = torch.randn((num_samples, *x.shape), device=x.device)

    def single(z_i):
        y, jvp_out = jvp(model, (x,), (z_i,))
        return (jvp_out * z_i).sum(-1), y

    div_samples, y_all = vmap(single)(z)
    return div_samples.mean(0), y_all[0]


def div_estimate(flow_model, diff_model, x, num_samples=2, alpha=0.8):
    """
    Calculate ∇·(v(x) * p(x)), where v(x) is the flow model and p(x) is
    represented implicitly by the score model.

    Args:
        flow_model: The flow model to compute the divergence for.
        score_model: The score model representing the probability density p(x).
        x: Input tensor for which to compute the divergence.
    """
    num_batch = x.shape[0]
    with torch.no_grad():
        s = -diff_model(x) / ((1 - alpha) ** 0.5) # [NOTE] ∇log(p(x)) = -Ɛ(x) / (1 - alpha_t).sqrt() according to the Tweedie equation
    div_term, v = hutchinson_estimate(flow_model, x, num_samples=num_samples)
    oth_term = torch.einsum('nd, nd -> n', v, s)

    output = {
        'div(pv)': div_term + oth_term,
        'score': s,
        'v': v,
    }
    return output

