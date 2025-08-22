# Pattern to Dynamics

This project learns the dynamics of Turing patterns using a score-based diffusion model and a flow model.

## How to Run

### 1. Generate the Dataset

First, you need to generate the Turing pattern dataset. `cd` into the `turing_pattern` directory and run the `generate_dataset.py` script.

```bash
cd turing_pattern
python generate_dataset.py --preset [waves] --cuda --normalize
```

The generated dataset will be saved in the `turing_pattern/data/` directory.

**Arguments:**

*   `--preset`: The type of Turing pattern to generate. This can be a single value or a list of values (e.g., `--preset waves spirals`). Choose from `waves`, `spirals`, `maze`, `life`.
*   `--num_samples`: Number of samples to generate (default: 8192).
*   `--height`, `--width`: Dimensions of the patterns (default: 128x128).
*   `--steps`: Number of simulation steps (default: 2000).
*   `--cuda`: Use GPU for simulation.
*   `--normalize`: Normalize the dataset. It is recommended to always use this flag.
*   `--chunk`: Chunk size for processing (default: 512).

### 2. Train the Score Model

Next, train the score-based diffusion model on the generated dataset.

```bash
python turing.py -m [waves] --wb -a 0.8
```

The trained model will be saved in the `turing_pattern/score_models/` directory, under a sub-directory named after the dataset.

**Arguments:**

*   `-m`, `--model`: The name of the dataset/model to use (e.g., `waves`).
*   `-n`, `--epochs`: Number of training epochs (default: 30).
*   `-b`, `--batch_size`: Batch size for training (default: 64).
*   `-l`, `--learning_rate`: Learning rate (default: 1e-3).
*   `--wb`: Use Weights & Biases for logging.
*   `-a`, `--alpha`: Alpha value for the diffusion model (default: 0.8).
*   `--warmup`: Number of warmup steps for the scheduler (default: 500).
*   `--plot_channel`: Channel to plot during training (default: 0).

### 3. Train the Flow Model

Finally, train the flow model to learn the dynamics.

```bash
python train_dynamics.py --dataset [waves] -a 0.8 --schedule --sym 0.1
```

The trained model will be saved as `turing_pattern/flow_models/[dataset_name].pth`.

**Arguments:**

*   `--dataset`: The dataset to use for training (e.g., `waves`).
*   `-a`, `--alpha`: Alpha value for the diffusion model (default: 0.8).
*   `-s`, `--schedule`: Use a cosine learning rate schedule.
*   `-g`, `--gaussian_weight`: Weight for the Gaussian term in the score model (default: 0.0).
*   `--sym`: Symmetry penalty weight for the loss function (default: 0.0).

## Configuration

The models and training parameters can be configured through the command-line arguments of the respective scripts.

*   **Dataset Generation:** `turing_pattern/generate_dataset.py`
*   **Score Model Training:** `turing.py`
*   **Flow Model Training:** `train_dynamics.py`
