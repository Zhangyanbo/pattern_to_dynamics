from PIL import Image, ImageColor, ImageChops
from tqdm.auto import tqdm
import random, math
from typing import Tuple, Optional, List


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

        return canvas, placed


if __name__ == "__main__":
    tiler = IconTiler("anchor.png")  # Requires PNG with alpha channel
    img, placed = tiler.generate(
        count=80,
        size=(1024, 1024),
        icon_scale=0.2, 
        bg_color="#FFFFFF",
        max_attempts_per_icon=2,
        verbose=True
    )
    img.save("./tiled.png")
