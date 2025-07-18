from dataset import GrayScottSimulator, create_random_state, TuringPatternDataset


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate Turing patterns dataset")
    parser.add_argument("--num_samples", type=int, default=8192, help="Number of samples to generate")
    parser.add_argument("--height", type=int, default=128, help="Height of the grid")
    parser.add_argument("--width", type=int, default=128, help="Width of the grid")
    parser.add_argument("--preset", type=str, default='waves', choices=['waves', 'life'], help="Preset pattern type")
    parser.add_argument("--cuda", action='store_true', help="Use CUDA for simulation")
    parser.add_argument("--steps", type=int, default=2000, help="Number of simulation steps")
    parser.add_argument("--chunk", type=int, default=512, help="Chunk size for processing")

    args = parser.parse_args()

    dataset = TuringPatternDataset(
        num_samples=args.num_samples,
        height=args.height,
        width=args.width,
        pattern_preset=args.preset,
        steps=args.steps,
        device='cuda' if args.cuda else 'cpu',
        chunk=args.chunk
    )

    # Save the dataset to a file
    dataset.save(os.path.join('data', f'{args.preset}_{args.height}x{args.width}.pt'))