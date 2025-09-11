from PIL import Image, ImageColor, ImageChops
from tqdm.auto import tqdm
import random, math
from typing import Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset
import os


class IconTiler:
    """
    Randomly place PNG icons with alpha channel on a given canvas:
    - Random rotation 0–360°
    - Pixel-level (alpha) non-overlapping
    - Wrapping boundaries (out-of-bounds parts appear from the other side)
    """

    def __init__(self, icon_path: str):
        self.icon = Image.open(icon_path).convert("RGBA")

    @staticmethod
    def _parse_color(c) -> Tuple[int, int, int]:
        if isinstance(c, (tuple, list)) and len(c) == 3:
            return tuple(int(v) for v in c)
        return ImageColor.getrgb(str(c))
    
    def convert(self, img, output_type):
        output_type = output_type.lower()
        if output_type == "pil":
            return img
        elif output_type == "numpy":
            return np.array(img).astype(np.float32) / 255.0
        elif output_type == "torch" or output_type == "pt":
            return torch.from_numpy(np.array(img)).float() / 255.0

    def generate(
        self,
        count: int,
        size: Tuple[int, int],
        bg_color: str | Tuple[int, int, int] = "white",
        *,
        icon_size: Optional[Tuple[int, int]] = None,  # e.g. (64, 64)
        icon_scale: Optional[float] = None,           # e.g. 0.5; mutually exclusive with icon_size (icon_size takes priority)
        max_attempts_per_icon: int = 2000,
        seed: Optional[int] = None,
        show_progress: bool = True,
        verbose: bool = False,
        output_type: str = "PIL" # "PIL", "numpy", "torch" / "pt"
    ) -> Image.Image:
        """
        Generate an image with randomly placed non-overlapping icons.
        
        Args:
            count: Number of icons to attempt to place on the canvas
            size: Canvas dimensions as (width, height) tuple
            bg_color: Background color as string (e.g. "white") or RGB tuple (255, 255, 255)
            icon_size: Explicit icon dimensions as (width, height). Takes priority over icon_scale
            icon_scale: Scale factor for resizing the original icon (e.g. 0.5 for half size)
            max_attempts_per_icon: Maximum placement attempts per icon before giving up
            seed: Random seed for reproducible results. If None, uses system random
            show_progress: Whether to display a progress bar during placement
            verbose: Whether to print verbose output
            output_type: Output type "PIL", "numpy", "torch" / "pt"
            
        Returns:
            canvas: PIL Image with placed icons. If unable to place all requested icons due to
                space constraints or max attempts reached, returns partial results with a
                warning message printed to console.
            placed: Number of icons placed on the canvas
            
        Note:
            - Icons are rotated randomly 0-360 degrees with bicubic resampling
            - Placement uses pixel-perfect alpha channel collision detection
            - Canvas boundaries wrap around (icons can span edges)
        """
        W, H = map(int, size)
        rng = random.Random(seed)

        # Canvas and occupancy mask
        canvas = Image.new("RGBA", (W, H), (*self._parse_color(bg_color), 255))
        occupied = Image.new("L", (W, H), 0)  # Binary occupancy (0/255)

        # Select base icon for this use (scalable)
        base_icon = self.icon
        if icon_size is not None:
            sw, sh = max(1, int(icon_size[0])), max(1, int(icon_size[1]))
            base_icon = self.icon.resize((sw, sh), Image.BICUBIC)
        elif icon_scale is not None:
            assert icon_scale > 0, "icon_scale must be > 0"
            ow, oh = self.icon.size
            sw, sh = max(1, int(round(ow * icon_scale))), max(1, int(round(oh * icon_scale)))
            base_icon = self.icon.resize((sw, sh), Image.BICUBIC)

        placements: List[Tuple[Image.Image, Tuple[float, float]]] = []
        attempts = 0
        placed = 0
        total_attempts_cap = max_attempts_per_icon * max(1, count)

        if verbose:
            pbar = tqdm(total=count, disable=not show_progress, desc="placing")
        else:
            pbar = None

        while placed < count and attempts < total_attempts_cap:
            attempts += 1

            # Random rotation
            angle = rng.uniform(0.0, 360.0)
            rotated = base_icon.rotate(angle, expand=True, resample=Image.BICUBIC)
            alpha = rotated.split()[-1]
            if not alpha.getbbox():
                continue

            rw, rh = rotated.size

            # Random center -> top-left (allow out-of-bounds, wrap around later)
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            tlx, tly = cx - rw / 2.0, cy - rh / 2.0

            # Build placement mask (consider wrapping: four translated copies)
            place_mask = Image.new("L", (W, H), 0)
            base_x = (tlx % W)
            base_y = (tly % H)
            for dx in (0, -W):
                for dy in (0, -H):
                    x = int(math.floor(base_x + dx))
                    y = int(math.floor(base_y + dy))
                    place_mask.paste(255, (x, y), alpha)

            # Pixel-level intersection detection (alpha domain)
            inter = ImageChops.multiply(place_mask, occupied)
            if inter.getbbox() is not None:
                continue  # Has overlap, abandon this attempt

            # Accept: update occupancy, record position
            occupied = ImageChops.lighter(occupied, place_mask)
            placements.append((rotated, (base_x, base_y)))
            placed += 1
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if placed < count:
            if verbose:
                print(f"[IconTiler] Only placed {placed}/{count} icons. Total attempts {attempts}."
                  f"Consider increasing canvas size, reducing count, or increasing max_attempts_per_icon.")

        # Draw to canvas (with four wrapped copies)
        for rotated, (x0, y0) in placements:
            for dx in (0, -W):
                for dy in (0, -H):
                    x = int(math.floor(x0 + dx))
                    y = int(math.floor(y0 + dy))
                    canvas.alpha_composite(rotated, (x, y))

        return self.convert(canvas, output_type), placed


class RandomImagesDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        num_icons: int,
        image_path: str,
        size: Tuple[int, int],
        bg_color: str | Tuple[int, int, int] = "white",
        *,
        icon_size: Optional[Tuple[int, int]] = None,  # e.g. (64, 64)
        icon_scale: Optional[float] = None,           # e.g. 0.5; mutually exclusive with icon_size (icon_size takes priority)
        max_attempts_per_icon: int = 2,
        show_progress: bool = True,
        normalization: bool = False,
        verbose: bool = False,
        precomputed_data: torch.Tensor = None,
    ):
        self.num_samples = num_samples
        self.image_path = image_path
        self.size = size
        self.bg_color = bg_color
        self.num_icons = num_icons
        self.icon_size = icon_size
        self.icon_scale = icon_scale
        self.max_attempts_per_icon = max_attempts_per_icon
        self.show_progress = show_progress
        self.verbose = verbose
        self.normalization = normalization
        self.arguments = dict(
            num_samples=num_samples,
            num_icons=num_icons,
            image_path=image_path,
            size=size,
            bg_color=bg_color,
            icon_size=icon_size,
            icon_scale=icon_scale,
            max_attempts_per_icon=max_attempts_per_icon,
            show_progress=show_progress,
            normalization=normalization,
            verbose=verbose,
        )

        if precomputed_data is not None:
            self.data = precomputed_data
        else:
            self.data = []
            tiler = IconTiler(self.image_path)
            for i in tqdm(range(self.num_samples), desc="Generating images"):
                img, placed = tiler.generate(
                    count=self.num_icons,
                    size=self.size,
                    icon_size=self.icon_size,
                    icon_scale=self.icon_scale,
                    bg_color=self.bg_color,
                    max_attempts_per_icon=self.max_attempts_per_icon,
                    show_progress=False,
                    verbose=False,
                    output_type="torch"
                ) # img: (H, W, 4)
                self.data.append(img.permute(2, 0, 1)[:3, ...])
            self.data = self.normalize(torch.stack(self.data))

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        save_data = dict(
            meta_data=self.arguments,
            data=self.data
        )
        torch.save(save_data, filepath)

    @classmethod
    def load(cls, filepath):
        save_data = torch.load(filepath)
        return cls(precomputed_data=save_data['data'], **save_data['meta_data'])

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def normalize(self, x):
        if self.normalization:
            return (x - 0.5) * 2
        return x

    def denormalize(self, x):
        if self.normalization:
            return (x + 1) / 2
        return x

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_icons", type=int, default=10)
    parser.add_argument("--image_path", type=str, default="./images/icon.png")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--bg_color", type=str, default="white")
    parser.add_argument("--icon_size", type=int, default=64)
    parser.add_argument("--max_attempts_per_icon", type=int, default=2)
    args = parser.parse_args()

    dataset = RandomImagesDataset(
        num_samples=args.num_samples,
        num_icons=args.num_icons,
        image_path=args.image_path,
        size=(args.size, args.size),
        bg_color=args.bg_color,
        icon_size=(args.icon_size, args.icon_size),
        max_attempts_per_icon=args.max_attempts_per_icon
    )
    dataset.save(os.path.join("data", f"random_images_{args.size}x{args.size}.pt"))