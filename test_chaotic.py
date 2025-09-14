from train_diffusion import Diffusion
from distributions import TwoPeaksDataset, RingDataset, LorenzDataset, TwoMoonsDataset
from train_dynamics import load_model
from models import FlowKernel, Flow
import torch
from torch.utils.data import DataLoader
from utils import integrate_rk4
import matplotlib.pyplot as plt
import numpy as np


def load_lorenz(id=0):
    score_model, dataset = load_model("lorenz")
    score_model.eval()
    flow_model = FlowKernel(dim=dataset.dim)
    flow_model.load_state_dict(
        torch.load(f"./results/lorenz/models/dynamics_model_{id}.pth")
    )
    flow_model.eval()
    return score_model.double(), flow_model.double(), dataset


def simulate_trajectory(f, x0, num_steps=5000, dt=0.1):
    trajectory = [x0]
    for i in range(num_steps - 1):
        with torch.no_grad():
            x0 = integrate_rk4(f, x0, dt)
        trajectory.append(x0)
    return torch.stack(trajectory, dim=0).cpu()


def filter_stable_trajectory(traj, threshold=4):
    max_corr = traj.reshape(traj.shape[0], -1).abs().max(dim=-1).values
    selected_idx = torch.where(max_corr < threshold)[0]
    traj_choosen = []
    for i in selected_idx:
        traj_choosen.append(traj[i])
    return traj_choosen


def get_stable_trajectory(
    f, num_traj, max_try=10, threshold=4, num_batch=32, num_steps=5000, dt=0.1, dim=3
):
    stable_traj = []
    try_count = 0
    while len(stable_traj) < num_traj:
        x0 = torch.randn(num_batch, dim, dtype=torch.float64)
        # traj.shape: (num_steps, num_batch, 3)
        traj = simulate_trajectory(f, x0, num_steps, dt).transpose(0, 1)
        traj = filter_stable_trajectory(traj, threshold=threshold)
        if len(traj) > 0:
            stable_traj.extend(traj)

        try_count += 1
        if try_count > max_try:
            break

    if len(stable_traj) > 0:
        stable_traj = stable_traj[:num_traj]
    else:
        print(f"No stable trajectory found")
    return stable_traj


def compute_distance(traj_perturbed, traj_ref):
    diff = traj_perturbed - traj_ref  # [num_steps, num_batch, dim]
    return diff.norm(dim=-1)  # [num_steps, num_batch]


def perturb_traj(reference_trajs, f, tau=500, sigma=1e-5, eps=1e-8, dt=0.1):
    n, T, dim = reference_trajs.shape
    num_simulation = T // tau
    distances = []
    x0 = reference_trajs[:, 0]
    dx = torch.randn_like(x0)
    dx = dx / dx.norm(dim=-1, keepdim=True) * sigma

    for i in range(num_simulation):
        traj_ref = reference_trajs[:, i * tau : (i + 1) * tau]
        x0 = traj_ref[:, 0]
        traj_perturbed = simulate_trajectory(
            f, x0 + dx, num_steps=tau, dt=dt
        ).transpose(0, 1)
        d = compute_distance(traj_perturbed, traj_ref)
        distances.append(d)
        dx = traj_perturbed[:, -1] - traj_ref[:, -1]
        dx = dx + torch.randn_like(dx) * eps  # avoid NaN
        dx = dx / dx.norm(dim=-1, keepdim=True) * sigma

    distances = torch.stack(distances, dim=0).mean(dim=0)
    return distances


def get_speed(traj, dt):
    # traj: [num_batch, num_steps, dim]
    diff = traj[:, 1:] - traj[:, :-1]  # [num_batch, num_steps-1, dim]
    movements = diff.norm(dim=-1).mean()
    speed = movements / dt
    return speed


def Lyapunov_exponent(
    f, num_traj=64, tau=500, total_steps=5000, sigma=1e-5, eps=1e-8, dt=0.1, dim=3
):
    stable_traj = get_stable_trajectory(f, num_traj, num_steps=total_steps, dim=dim)
    if len(stable_traj) > 0:
        reference_trajs = torch.stack(stable_traj, dim=0)  # [num_traj, num_steps, dim]
    else:
        # no stable trajectory found
        return None, None

    distances = perturb_traj(reference_trajs, f, tau, sigma, eps, dt)
    speed = get_speed(reference_trajs, dt)

    simulation_data = {
        "reference_trajs": reference_trajs,
        "distances": distances,
        "speed": speed,
    }

    distances = distances.mean(dim=0).log()
    exponent = (distances[-1] - distances[0]) / (tau * dt)
    normalized_exponent = (distances[-1] - distances[0]) / speed

    return exponent.item(), normalized_exponent.item(), simulation_data


def save_results(exponents, exponents_normalized, data):
    import pickle
    import json

    path = "./results/lorenz/lyapunov/lyapunov_exponent.json"
    with open(path, "w") as f:
        json.dump(
            {"exponents": exponents, "exponents_normalized": exponents_normalized},
            f,
            indent=4,
        )

    path = "./results/lorenz/lyapunov/lyapunov_exponent.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "exponents": exponents,
                "exponents_normalized": exponents_normalized,
                "data": data,
            },
            f,
        )


def save_meta_data(args):
    import json

    path = "./results/lorenz/lyapunov/lyapunov_meta_data.json"
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=4)


def main(args):
    from tqdm import tqdm

    num_models = args.num_models

    if args.specific_model != -1:
        models = [args.specific_model]
    else:
        models = list(range(num_models))

    exponents = []
    exponents_normalized = []
    data = []
    for i in tqdm(models):
        score_model, flow_model, dataset = load_lorenz(i)
        if args.score_ratio > 0:
            r = args.score_ratio
            f = lambda x: score_model.score(x, t=0.1) * (r**0.5) + flow_model(x) * (
                (1 - r) ** 0.5
            )
        else:
            f = flow_model
        exponent, exponent_normalized, simulation_data = Lyapunov_exponent(
            f,
            num_traj=args.num_traj,
            tau=args.tau,
            total_steps=args.total_steps,
            sigma=args.sigma,
            eps=args.eps,
            dt=args.dt,
        )
        tqdm.write(
            f"Model {i} has Lyapunov exponent {exponent:.4f} (normalized: {exponent_normalized:.4f})"
        )
        exponents.append(exponent)
        exponents_normalized.append(exponent_normalized)
        data.append(simulation_data)

    save_results(exponents, exponents_normalized, data)


if __name__ == "__main__":
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_models", "-n", type=int, default=100)
    parser.add_argument("--specific_model", "-m", type=int, default=-1)
    parser.add_argument("--num_traj", "-s", type=int, default=64)
    parser.add_argument("--tau", "-t", type=int, default=2500)
    parser.add_argument("--total_steps", "-T", type=int, default=5000)
    parser.add_argument("--sigma", "-sigma", type=float, default=1e-5)
    parser.add_argument("--eps", "-eps", type=float, default=1e-20)
    parser.add_argument("--dt", "-dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score_ratio", "-r", type=float, default=0.0)

    args = parser.parse_args()

    # initialize random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    save_meta_data(args)
    main(args)
