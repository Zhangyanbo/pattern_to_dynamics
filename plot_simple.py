import numpy as np
import matplotlib.pyplot as plt
import torch
from train_dynamics import load_model
from models import FlowKernel, Flow
from matplotlib.lines import Line2D

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

BACKGROUND_COLOR = '#CECECE'
TRAJECTORY_COLOR = '#9BE1E1'
DATA_COLOR = '#628FCE'


def load_trained_model(model:str, id:int=0):
    score_model, dataset = load_model(model)
    score_model.eval()
    flow_model = FlowKernel(dim=dataset.dim)
    flow_model.load_state_dict(torch.load(f'./results/{model}/models/dynamics_model_{id}.pth'))
    flow_model.eval()
    return score_model, flow_model, dataset

def plot_flow_streamlines(flow_model, plot_range=(-3, 3), grid_size=20, ax=None, color=TRAJECTORY_COLOR, stream_args={}):
    """
    Create a streamplot visualization of a flow model's vector field.
    
    Args:
        flow_model: The flow model to visualize
        plot_range: Tuple (min, max) for both x and y axes
        grid_size: Number of grid points in each dimension
        ax: Matplotlib axes to plot on. If None, creates a new figure
    """
    # Create a grid for the vector field
    x = torch.linspace(plot_range[0], plot_range[1], grid_size)
    y = torch.linspace(plot_range[0], plot_range[1], grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Get numpy arrays for streamplot
    x_np = x.numpy()
    y_np = y.numpy()
    u = np.zeros((grid_size, grid_size))
    v = np.zeros((grid_size, grid_size))
    
    # Calculate vector field at each point
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    with torch.no_grad():
        vectors = flow_model(points).cpu()
    
    # Reshape vectors for streamplot
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            u[i, j] = vectors[idx, 0].item()
            v[i, j] = vectors[idx, 1].item()
    
    # Create or use provided axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create streamplot with 1D arrays for x and y
    # Use linewidth to control opacity instead of alpha
    ax.streamplot(x_np, y_np, u.T, v.T, color=color, **stream_args)  # Transpose u,v to match x,y coordinates
    ax.set_xlim(plot_range)
    ax.set_ylim(plot_range)
    
    return ax

def plot_random_points(dataset, n_samples=256, color=DATA_COLOR, alpha=0.75, ax=None):
    """
    Plot randomly sampled points from a dataset's time series.
    
    Args:
        dataset: Dataset containing time_series data
        n_samples: Number of points to sample (default: 256)
        color: Color for the points (default: '#FBBCA2')
        alpha: Transparency of points (default: 0.75)
        ax: Matplotlib axes to plot on. If None, uses current axes
    """
    # Sample random points from the dataset
    points = dataset.time_series[np.random.choice(len(dataset.time_series), n_samples, replace=False)]
    
    # Use provided axes or get current axes
    if ax is None:
        ax = plt.gca()
    
    # Plot the points
    ax.plot(points[:, 0], points[:, 1], '.', color=color, markeredgewidth=0, alpha=alpha)
    
    return ax

def create_dataset_plot(dataset_name, ax):
    """Create a plot for a specific dataset with flow streamlines and random points."""
    score_model, flow_model, dataset = load_trained_model(dataset_name, id=0)
    ax.set_aspect('equal')
    plot_flow_streamlines(flow_model, ax=ax, color='#9BE1E1', stream_args={'linewidth': 1., 'density': 0.8})
    plot_random_points(dataset, ax=ax, color='#628FCE', alpha=0.75)
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    return flow_model, dataset

def create_legend_plot(ax):
    """Create a legend plot showing what the colors represent."""
    # Create dummy plots for legend
    ax.set_aspect('equal')
    # ax.set_title('Legend')
    
    # Create dummy elements for the legend
    blue_line = Line2D([0], [0], color='#9BE1E1', lw=2)
    blue_dot = Line2D([0], [0], marker='o', color='w', markerfacecolor='#628FCE', markersize=8)
    
    # Add legend
    ax.legend([blue_line, blue_dot], ['Flow streamlines', 'Data points'], loc='center')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def setup_nested_grid(ax, nrows=2, ncols=2, use_3d=False, wspace=0.3, hspace=0.3):
    """
    Set up a nested grid within the provided axes using GridSpec,
    without removing the parent Axes (so we can still use its title, etc).
    """
    fig = ax.figure
    pos = ax.get_position()

    # Hide the parent ax, but keep it alive
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Now build the nested axes using GridSpec manually placed inside `ax`'s bounds
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    gs.update(left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1, wspace=wspace, hspace=hspace)

    axs = []
    for idx, (i, j) in enumerate([(i, j) for i in range(nrows) for j in range(ncols)]):
        if use_3d and idx == 0:
            axs.append(fig.add_subplot(gs[i, j], projection='3d'))
        else:
            axs.append(fig.add_subplot(gs[i, j]))

    return axs


def create_multi_dataset_plot(ax=None, figsize=(5, 5), wspace=0.1, hspace=0.1):
    """
    Create a 2x2 subplot with three datasets and a legend.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes, optional
        Parent axes to nest the 2x2 grid into. If None, a new figure is created.
    figsize : tuple, optional
        Figure size if creating a new figure.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plots.
    axs : array of matplotlib.axes.Axes
        The axes containing the plots.
    """
    if ax is None:
        # Create a new figure with 2x2 grid
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()
    else:
        # Create a nested 2x2 grid within the provided axes
        fig = ax.figure
        axs = setup_nested_grid(ax, 2, 2, wspace=wspace, hspace=hspace)
    
    # Plot the three datasets
    create_dataset_plot('two_peaks', axs[0])
    create_dataset_plot('two_moons', axs[1])
    create_dataset_plot('ring', axs[2])
    
    # Create legend in the fourth subplot
    create_legend_plot(axs[3])
    
    if ax is None:
        plt.tight_layout()
    
    return fig, axs

def plot_2d_projection(results, dataset, idx, i=0, j=1, num_points=5000, 
                       traj_color=TRAJECTORY_COLOR, traj_linewidth=1.5, 
                       background_color=BACKGROUND_COLOR, point_alpha=0.75, 
                       axes_on_right=False,
                       ax=None, show=True):
    """
    Plot a 2D projection of the trajectory and dataset points.
    
    Args:
        results: Dictionary containing results data
        dataset: Dataset object containing time series data
        idx: Index of the trajectory to plot
        i: First dimension index (default: 0)
        j: Second dimension index (default: 1)
        num_points: Number of points to plot from dataset (default: 5000)
        traj_color: Color for trajectory line (default: '#9BE1E1')
        traj_linewidth: Width of trajectory line (default: 1.5)
        point_color: Color for dataset points (default: '#FBBCA2')
        point_alpha: Alpha transparency for points (default: 0.75)
        ax: Matplotlib axis to plot on (default: None, creates new axis)
        show: Whether to call plt.show() (default: True)
        
    Returns:
        fig, ax: Figure and axis objects
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Get trajectory and plot
    traj = results['data'][idx]['reference_trajs'][0]
    ax.plot(traj[:, i], traj[:, j], color=traj_color, linewidth=traj_linewidth, zorder=1, label='Learned', alpha=0.75)
    
    # Plot dataset points
    ax.plot(dataset.time_series[:num_points, i], dataset.time_series[:num_points, j], 
            alpha=point_alpha, color=background_color, label='Lorenz', zorder=0)
    
    # Add labels
    ax.set_xlabel(f'$x_{i+1}$')
    
    # Position y-label on left or right based on i and j
    if axes_on_right:
        # Place y-label on right side
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='y', labelright=True, labelleft=False)
        ax.yaxis.tick_right()
    else:
        # Default: place y-label on left side
        ax.yaxis.set_label_position("left")
        ax.tick_params(axis='y', labelleft=True, labelright=False)
        ax.yaxis.tick_left()
    
    ax.set_ylabel(f'$x_{j+1}$')

    # set aspect ratio to 1
    ax.set_box_aspect(1)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    
    if show:
        plt.show()
    
    return fig, ax

def plot_lyapunov_exponent(results, idx, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
        
    ax.plot(results['data'][idx]['distances'].mean(dim=0), color=DATA_COLOR)
    ax.semilogy()
    ax.set_xlabel('Time step')
    ax.set_ylabel('Distance')
    
    # Set 1:1 box aspect using data limits
    ax.set_box_aspect(1)
    return fig, ax

def plot_lorenz_combined(results, dataset, idx, num_points=5000, ax=None, wspace=0.2, hspace=0.2):
    """
    Create a 2x2 plot with a 3D plot and three 2D projections of the Lorenz system.
    
    Args:
        results: Dictionary containing results data
        dataset: Dataset object containing time series data
        idx: Index of the trajectory to plot
        num_points: Number of points to plot from dataset (default: 5000)
        ax: Matplotlib axis to plot on (default: None, creates new figure)
        
    Returns:
        fig: The figure object containing the subplots
    """
    # Create figure or use existing axis
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        is_nested = False
    else:
        fig = ax.figure
        is_nested = True
    
    # Create the appropriate axes based on whether we're in nested mode
    axes = create_appropriate_axes(fig, ax, is_nested, wspace, hspace)
    plot_lyapunov_exponent(results, idx, ax=axes[0])
    create_2d_projections(axes[1:], results, dataset, idx, num_points)
    
    return fig

def create_appropriate_axes(fig, parent_ax, is_nested, wspace=0.2, hspace=0.2):
    """
    Create the appropriate axes arrangement based on whether we're in nested mode.
    
    Args:
        fig: The figure to add the subplots to
        parent_ax: The parent axis (if in nested mode)
        is_nested: Whether we're creating a nested plot
        wspace: Width space between subplots
        hspace: Height space between subplots
        
    Returns:
        list: List of axes for plotting (first is 3D, others are 2D)
    """
    if not is_nested:
        # Create standalone 2x2 grid
        ax_2d = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
        return ax_2d
    else:
        # Create nested grid within parent_ax
        return setup_nested_grid(parent_ax, wspace=wspace, hspace=hspace)

from matplotlib import gridspec

def create_2d_projections(axes, results, dataset, idx, num_points):
    """
    Create the three 2D projections for the combined plot.
    
    Args:
        axes: List of axes to plot on
        results: Dictionary containing results data
        dataset: Dataset object containing time series data
        idx: Index of the trajectory to plot
        num_points: Number of points to plot from dataset
    """
    # Define the dimension pairs for each projection
    projections = [(0, 1), (0, 2), (1, 2)]
    y_axes_on_right = [True, False, True]
    legend_added = False
    
    # Create each 2D projection
    for ax, (i, j), axes_on_right in zip(axes, projections, y_axes_on_right):
        plot_2d_projection(results, dataset, idx, i=i, j=j, 
                          num_points=num_points, ax=ax, show=False, axes_on_right=axes_on_right)
        if not legend_added:
            ax.legend()
            legend_added = True

if __name__ == '__main__':
    # set random seed
    np.random.seed(0)
    torch.manual_seed(0)
    import pickle
    # Nested usage example
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    create_multi_dataset_plot(ax=axs[0])
    axs[0].set_title('(A) 2d systems')

    results = pickle.load(open('./results/lorenz/lyapunov_exponent.pkl', 'rb'))
    idx = 61
    score_model, flow_model, dataset = load_trained_model('lorenz', id=idx)
    fig = plot_lorenz_combined(results, dataset, idx, ax=axs[1], wspace=0.2, hspace=0.1)
    axs[1].set_title('(B) Lorenz system')
    plt.savefig('./results/simple_systems.png', dpi=300, bbox_inches='tight')
    plt.savefig('./results/simple_systems.pdf', bbox_inches='tight')
    plt.close()