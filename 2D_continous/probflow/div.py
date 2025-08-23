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


def div_estimate(flow_model, score_model, x, num_samples=2, x0=None):
    """
    Calculate ∇·(v(x) * p(x)), where v(x) is the flow model and p(x) is
    represented implicitly by the score model.

    Args:
        flow_model: The flow model to compute the divergence for.
        score_model: The score model representing the probability density p(x).
        x: Input tensor for which to compute the divergence.
    
    Outputs:
        A dictionary containing:
            - 'div(pv)_1': First term of the divergence estimate. Shape: [batch_size].
            - 'div(pv)_2': Second term of the divergence estimate. Shape: [batch_size].
            - 'div(v)': Divergence of the flow model v(x). Shape: [batch_size].
            - 'v@s': Dot product of v(x) and the score s(x). Shape: [batch_size].
            - 'score': The score s(x) = ∇ log(p(x)). Shape: [batch_size, dim].
            - 'v': The flow model output v(x). Shape: [batch_size, dim].
    """
    num_batch = x.shape[0]
    with torch.no_grad():
        s = score_model(x)
    
    # Compute the divergence estimate twice to get unbiased estimates for both values and gradients
    if x0 is None:
        # If x0 is provided, we use it to compute the divergence
        div_term_1, v = hutchinson_estimate(flow_model, x, num_samples=num_samples // 2)
        div_term_2, _ = hutchinson_estimate(flow_model, x, num_samples=num_samples // 2)
    else:
        # Otherwise, we use the diffused version of x, i.e., x_t
        div_term_1, v = hutchinson_estimate(flow_model, x0, num_samples=num_samples // 2)
        div_term_2, _ = hutchinson_estimate(flow_model, x0, num_samples=num_samples // 2)
    oth_term = torch.einsum('nd, nd -> n', v, s)

    output = {
        'div(pv)_1': div_term_1 + oth_term,
        'div(pv)_2': div_term_2 + oth_term,
        'div(v)': div_term_1,
        'v@s': oth_term,
        'score': s,
        'v': v,
    }
    return output

