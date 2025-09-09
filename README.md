# Pattern to Dynamics

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