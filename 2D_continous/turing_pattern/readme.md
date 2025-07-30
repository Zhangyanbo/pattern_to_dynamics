# Generating Dataset of Gray-Scott Model

By running the following code, the program will simulate Gray-Scott Model, and save the pre-computed dataset at `./data/`:

```bash
python generate_dataset.py --preset life maze waves spirals --cuda --normalize
```

This will simulate 8,192 randomly simulated Gray-Scott model patterns, and save the result at `./data/[preset_name]_hxw.pt`.

## Parameters

- `--preset`: Pattern type to generate. Available options: `waves`, `life`, `spots`, `stripes`, `spirals`, `holes`, `chaos`, `maze`, `bubbles`, `worms`, `solitons`, `dots`
- `--cuda`: Use GPU acceleration (optional)
- `--num_samples`: Number of samples to generate (default: 8192)
- `--height`, `--width`: Grid dimensions (default: 128x128)
- `--steps`: Simulation steps (default: 2000)

## Examples

Generate waves pattern:
```bash
python generate_dataset.py --preset waves
```

Generate life pattern with GPU:
```bash
python generate_dataset.py --preset life --cuda
```

## Output

The generated dataset contains 2D reaction-diffusion patterns simulated using the Gray-Scott model. Each pattern is a continuous-valued 2D array representing the concentration of chemical species over time.