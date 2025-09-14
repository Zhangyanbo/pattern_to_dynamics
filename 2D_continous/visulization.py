import matplotlib.pyplot as plt
import torch


class TuringPlotter:
    RANGES = {
        "life": [(0.13, 1), (0, 0.33)],
        "waves": [(0.12, 1), (0, 0.42)],
        "spirals": [(0.05, 0.7), (0, 0.37)],
        "maze": [(0.2, 0.8), (0, 0.35)],
    }

    def __init__(self):
        self.colors = ["#FFFFFF", "#628FCE"]
        self.background = "#9BE1E1"  #'#F6F1F1'

        self.color_rgb = [
            list(int(color[i : i + 2], 16) for i in (1, 3, 5)) for color in self.colors
        ]
        self.color_rgb = torch.Tensor(self.color_rgb) / 255
        self.color_back = list(int(self.background[i : i + 2], 16) for i in (1, 3, 5))
        self.color_back = torch.Tensor(self.color_back) / 255

    def plot(self, img, name, ax=None):
        if ax is None:
            ax = plt.gca()
        ranges = self.RANGES[name]
        img01 = torch.stack(
            [(img[0, i] - m) / (M - m) for i, (m, M) in enumerate(ranges)]
        )
        img01_back = torch.clip(1 - img01.permute(1, 2, 0).norm(dim=-1), 0, 1)  # [H, W]
        imgRGB = torch.einsum("chw, cR -> Rhw", img01.cpu(), self.color_rgb)
        imgRGB += torch.einsum("hw, R -> Rhw", img01_back.cpu(), self.color_back)
        imgRGB = imgRGB.permute(1, 2, 0)

        ax.imshow(imgRGB.clip(0, 1))


def plot_trace(trajectory, dt, num_rows, num_cols, name):
    plotter = TuringPlotter()

    for i, t in enumerate(range(0, (num_rows * num_cols) * dt, dt)):
        plt.subplot(num_rows, num_cols, i + 1)
        plotter.plot(trajectory[min(t, len(trajectory) - 1)], name)
        # remove title and axis
        plt.title(f"Time {t}")
        plt.axis("off")
