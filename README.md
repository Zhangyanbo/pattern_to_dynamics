# *Equilibrium flow*: From Snapshots to Dynamics

How snapshot distribution constraints the possible dynamics? When we see a pattern, how confidently can we say "This is the underlying dynamics" without seeing the time evolution? How artificial life relates to real biological lifes? To answer these fundamental question, we propose ***Equilibrium flow***: by learning the distribution-preserving dynamics, we can find possible dynamics to preserve the given data distribution without time information.

![cover](./media/cover.png)

For **2D systems**, our method finds interesting non-trivial dynamics that preserve them.
For **Lorenz system**, a dynamical system with chaotic behavior, the recovered dynamics also exhibit chaotic behavior with positive Lyapunov exponents. For **Turing patterns**, we propose a training-free method, which has limited solution space, but much faster. The resulted dynamics also highly aligned to the ground-truth.

Beyond these, we also explore the design capability with our method on Artificial Lifes. With given manuual designed patterns, our method not only finds the dynamics / neural cellular automata that preserves the pattern, but also see collective behaviors.

# Running model
## Lorenz Systems

### Step 1: Train diffusion model

Run the following command:

```bash
python train_diffusion.py --model lorenz
```

This will train a diffusion model on the Lorenz system. The trained model is saved in `./results/lorenz/diffusion_model.pth`.

### Step 2: Train dynamics model

Run:

```bash
python train_dynamics.py --model lorenz --num_experiments 1
# You can set num_experiments to the desired number of experiments if you want multiple results
```

The trained dynamics model is saved in `./results/lorenz/models/dynamics_models_<id>.pth`.

### Loading model