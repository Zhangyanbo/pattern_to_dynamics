import matplotlib
import matplotlib.pyplot as plt

def plot_gol_comparison(x_gol, x_sampled):
    """
    Plot Game of Life ground truth and diffusion sampled states side by side.
    
    Args:
        x0: Ground truth Game of Life state tensor
        x_t: Diffusion sampled state tensor
    """
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    dot_size = 1

    def plot_state(state, subplot_idx, title):
        plt.subplot(1, 2, subplot_idx)
        scatter = state.cpu().numpy()
        for i in range(64):
            for j in range(64):
                if scatter[i, j] > 0.5:
                    plt.scatter(i, j, color='black', marker='s', s=dot_size)
        plt.title(title)
        plt.gca().set_aspect('equal', 'box')
        plt.xlim(-1, 65)
        plt.ylim(-1, 65)
        plt.xticks([])
        plt.yticks([])

    plot_state(x_gol, 1, '(A) Game of Life')
    plot_state(x_sampled, 2, '(B) Diffusion Sampled')

    plt.savefig('./figures/GoL_diffusion_sample_and_ground_truth.png', bbox_inches='tight')
    plt.savefig('./figures/GoL_diffusion_sample_and_ground_truth.pdf', bbox_inches='tight')
    plt.close()