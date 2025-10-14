import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from torch.nn import Module
from torch.utils.data import Dataset


def plot_decision_boundary(network: Module, dataset: Dataset):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a meshgrid over the feature space
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(next(network.parameters()).device)

    # Model prediction
    network.eval()
    with torch.no_grad():
        probs = network({"input": grid_tensor})["output"]
        Z = probs[:, 1]  # probability of class 1 (= 1 - probability of class 0)
        Z = Z.cpu().numpy().reshape(xx.shape)

    # Plot filled contour
    contour = ax.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6, vmin=0, vmax=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)  # decision boundary
    ax.set_title("Decision Boundary")

    # Plot original data points for each class separately so they can have legend entries
    class0 = dataset.labels == 0
    class1 = dataset.labels == 1
    ax.scatter(
        dataset.data[class0, 0],
        dataset.data[class0, 1],
        c="blue",
        edgecolors="k",
        alpha=0.8,
        label="Class A",
    )
    ax.scatter(
        dataset.data[class1, 0],
        dataset.data[class1, 1],
        c="red",
        edgecolors="k",
        alpha=0.8,
        label="Class B",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add hatched region
    hatched_area = Rectangle(
        (0, 0),  # bottom-left corner
        0.25,  # width
        1,  # height
        facecolor="none",
        edgecolor="green",
        hatch="//",
        linewidth=1,
        alpha=0.7,
        label="Constrained region",
    )
    ax.add_patch(hatched_area)

    # Colorbar
    sm = ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Probability of class B")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])

    # Add legend
    ax.legend(loc="upper right", frameon=True)

    return ax
