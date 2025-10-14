import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torch.nn import Module
from torch.utils.data import Dataset


def plot_regression_epoch(network: Module, dataset: Dataset, device: torch.device):
    fig, ax = plt.subplots(figsize=(8, 6))

    network.eval()
    with torch.no_grad():
        preds = network({"input": dataset.inputs.to(device)})["output"].cpu().numpy().flatten()

    # Plot noisy observations
    ax.scatter(
        dataset.inputs.numpy(),
        dataset.targets.numpy(),
        color="gray",
        alpha=0.6,
        edgecolors="k",
        s=25,
        label="Input data",
    )

    # Plot model prediction
    ax.plot(dataset.inputs.numpy(), preds, "r-", linewidth=2, label="Prediction")

    # Add hatched region
    hatched_area = Rectangle(
        (0.3, 0),  # bottom-left corner
        0.3,  # width
        1,  # height
        facecolor="none",
        edgecolor="green",
        hatch="//",
        linewidth=1,
        alpha=0.7,
        label="Constrained region",
    )
    ax.add_patch(hatched_area)

    # Limit axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Titles and labels
    ax.set_title("Model predictions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    return ax
