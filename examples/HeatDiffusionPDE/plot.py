import os

import numpy as np
from config import CGGDConfig
from data import Loader
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import Module
from torch.utils.data import (
    TensorDataset,
)

from congrads.callbacks.base import Callback
from congrads.utils.utility import DictDatasetWrapper


def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class PlotCallback(Callback):
    def __init__(self, config: CGGDConfig, network: Module, loader: Loader, device="cpu"):
        super().__init__()

        self.config = config
        self.network = network
        self.loader = loader
        self.device = device

        X, u, c = self.loader.regular_grid(N=101)  # pre-generate grid for plotting

        dataset = TensorDataset(X, u, c)
        dataset = DictDatasetWrapper(dataset, ["input", "target", "context"])
        self.data_loader = self.loader._dataset_to_loader(
            dataset, batch_size=len(dataset), shuffle=False
        )

        os.makedirs("plots", exist_ok=True)

    def on_epoch_end(self, data, ctx):
        super().on_epoch_end(data, ctx)
        epoch = data["epoch"]

        if epoch % 50 != 0:
            return

        self.network.eval()
        batch = next(iter(self.data_loader))

        # Move to network device
        batch = {key: batch[key].to(self.device) for key in batch.keys()}

        # Forward pass
        batch = self.network(batch)

        # Move to CPU for plotting
        batch = {key: batch[key].cpu() for key in batch.keys()}

        self._plot(epoch, batch)

    def _plot(self, epoch, preds):
        # Convert to numpy
        X, u_true, u_pred = (
            to_numpy(preds["input"]),
            to_numpy(preds["target"]),
            to_numpy(preds["output"]),
        )

        # Reshape to 2D grid
        N = int(np.sqrt(len(X)))
        x = np.unique(X[:, 0])
        t = np.unique(X[:, 1])

        u_true_grid = u_true.reshape(N, N).T
        u_pred_grid = u_pred.reshape(N, N).T
        u_diff_grid = np.abs(u_pred_grid - u_true_grid)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True, constrained_layout=True)

        # --- True solution ---
        im0 = axes[0].imshow(
            u_true_grid, extent=[t.min(), t.max(), x.min(), x.max()], origin="lower", aspect="auto"
        )
        axes[0].set_title("True Solution")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("x")

        # --- Predicted solution ---
        im1 = axes[1].imshow(
            u_pred_grid, extent=[t.min(), t.max(), x.min(), x.max()], origin="lower", aspect="auto"
        )
        axes[1].set_title("Predicted Solution")
        axes[1].set_xlabel("t")

        cbar1 = fig.colorbar(im1, ax=axes[1])
        cbar1.set_label("u")

        # --- Difference ---
        im2 = axes[2].imshow(
            u_diff_grid, extent=[t.min(), t.max(), x.min(), x.max()], origin="lower", aspect="auto"
        )
        axes[2].set_title("Difference")
        axes[2].set_xlabel("t")

        cbar2 = fig.colorbar(im2, ax=axes[2])
        cbar2.set_label("|û - u|")

        fig.suptitle("True vs Predicted Solutions")

        # Save images
        plt.savefig("plots/SimplePDE.png")
        plt.close(fig)
