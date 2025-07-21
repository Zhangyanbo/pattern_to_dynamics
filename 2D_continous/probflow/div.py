import torch


def _hutchinson_estimate(vjp_func, x, z):
    vjp = vjp_func(z)[0]
    return torch.einsum('nd, nd -> n', vjp, z)

def hutchinson_estimate(model, x, num_samples=2):
    """
    Calculate ∇·model(x) using Hutchinson's trace estimator.

    Args:
        model: The model to compute the gradient for.
        x: Input tensor for which to compute the gradient.
        num_samples: Number of samples to use for the estimation.
    
    Returns:
        ∇·model(x): The estimated divergence of the model at x.
        model(x): The model's output at x.
    
    [NOTE] The x input should be a tensor of shape (batch_size, dim).
    When dealing with 2D or other higher-dimensional data, ensure
    that the input is flattened appropriately before passing it to 
    this function. Accordingly, the model should also be designed to handle
    such flattened inputs.
    """
    output, vjp_func= torch.func.vjp(model, x)
    z = torch.randn((num_samples, *x.shape), device=x.device)
    est = torch.vmap(_hutchinson_estimate, (None, None, 0))(vjp_func, x, z)
    return est.mean(dim=0), output

def div_estimate(flow_model, diff_model, x, num_samples=2):
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
        s = -diff_model(x) # [NOTE] ∇log(p(x)) = -Ɛ(x) / (1 - alpha_t).sqrt() according to the Tweedie equation
    div_term, v = hutchinson_estimate(flow_model, x, num_samples=num_samples)
    oth_term = torch.einsum('nd, nd -> n', v, s)

    output = {
        'div(pv)': div_term + oth_term,
        'score': s,
        'v': v,
    }
    return output

