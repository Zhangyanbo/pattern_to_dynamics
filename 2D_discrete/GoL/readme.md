# Game of Life Experiment

## Train Score Function Model

Run `train_diffusion.py --mode train --num_epochs 500` to train a diffusion model on simulated Game of Life data. The trained model is saved in `./models/`, with the `GoL_diffusion.pth` file for weights, `GoL_diffusion.json` file for model configuration, and `losses.json` for the loss record.

The trained model can be loaded by:

```python
import json
from diffusers import UNet2DModel


with open('./models/GoL_diffusion.json', 'r') as f:
    config = json.load(f)
unet = UNet2DModel(**config).cuda()
unet.load_state_dict(torch.load('./models/GoL_diffusion.pth'))
```

## Supporting functions

### `GameOfLife`: `GoLData.py`

The `GameOfLife` class implements a fast PyTorch-based Game of Life simulator using convolution operations.

**Key Method:**
`.forward(state, steps=1)`: Evolves the Game of Life from initial `state` for `steps` evolution steps
- Input: `state` tensor of shape `[batch_size, 1, height, width]` with binary values (0/1)
- Output: tensor of shape `[batch_size, steps+1, 1, height, width]` containing all states including initial

### `GoLDataset`: `GoLData.py`

A PyTorch Dataset for generating Game of Life training data. Creates samples by evolving random initial states for a specified number of steps.

**Key Parameters:**
- `num_samples`: Number of samples in the dataset
- `height`, `width`: Dimensions of the Game of Life grid
- `steps`: Number of evolution steps to simulate
- `mode`: Either `"precompute"` (generate all samples upfront) or `"online"` (generate on-the-fly)

**Output:** Each sample is a tensor of shape `[1 (channel), height, width]` representing the final state after evolution.