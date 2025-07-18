import torch
import matplotlib.pyplot as plt
import matplotlib

def plot_eca_sorted(x_real, x_sampled, *, savepath='figures/ECA_sorted', dpi=300):
    """
    Compare ground‑truth and generated 1‑D ECA patterns by visualising
    them after sorting each set lexicographically (binary‑integer order).

    Parameters
    ----------
    x_real : Tensor  (N, W) or (N, 1, W)  -- reference patterns
    x_sampled : Tensor  (N, W) or (N, 1, W)  -- generated patterns
    savepath : str  -- prefix for PNG/PDF outputs (no extension)
    dpi : int  -- DPI for saved figures
    """
    # --- prepare tensors ----------------------------------------------------
    x_real     = x_real.squeeze(1).float()   # (N, W)
    x_sampled  = x_sampled.squeeze(1).float()
    assert x_real.ndim == x_sampled.ndim == 2, "Expect shape (N, W) or (N,1,W)"

    # width check for 64‑bit conversion
    W = x_real.shape[1]
    if W > 64:
        raise ValueError("Width >64 needs string‑based sort; see code comment.")

    # turn each pattern into an integer key
    powers = 2 ** torch.arange(W - 1, -1, -1, device=x_real.device)  # MSB‑first
    def sort_rows(x):
        keys = (x * powers).sum(-1).long()      # integer key
        order = torch.argsort(keys)
        return x[order]

    x_real_sorted    = sort_rows(x_real)
    x_sampled_sorted = sort_rows(x_sampled)

    # --- plotting -----------------------------------------------------------
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    fig, axs = plt.subplots(1, 2, figsize=(6, 6), sharey=True)

    axs[0].imshow(x_real_sorted.cpu(), cmap='Greys', aspect='auto', interpolation='nearest')
    axs[0].set_title('(A) ECA Ground Truth')
    axs[1].imshow(x_sampled_sorted.cpu(), cmap='Greys', aspect='auto', interpolation='nearest')
    axs[1].set_title('(B) ECA Sampled')

    for ax in axs:
        ax.set_xlabel('Cell index')
        ax.set_ylabel('Pattern rank')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f'{savepath}.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(f'{savepath}.pdf', dpi=dpi, bbox_inches='tight')
    plt.close()

################################################################################
# Usage example
#
# real    = torch.randint(0, 2, (256, 1, 64)).float()
# sampled = torch.randint(0, 2, (256, 1, 64)).float()
# plot_eca_sorted(real, sampled)
################################################################################
