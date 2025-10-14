import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Subset

from congrads.descriptor import Descriptor


def get_all_data(subset):
    """Extract and stack all data from a PyTorch Dataset or Subset that returns dictionaries of tensors (with matching shapes).

    Args:
        subset (torch.utils.data.Subset or Dataset): Dataset or subset to extract from.

    Returns:
        dict[str, torch.Tensor]: Dictionary with each key stacked along dimension 0.
    """
    if isinstance(subset, Subset):
        dataset, indices = subset.dataset, subset.indices
        samples = [dataset[i] for i in indices]
    else:
        samples = [subset[i] for i in range(len(subset))]

    return {key: torch.stack([s[key] for s in samples]) for key in samples[0].keys()}


def plot_regression_epoch(
    descriptor: Descriptor,
    network: Module,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
):
    """Plot model predictions versus signal for training and test waveforms.

    Args:
        descriptor (Descriptor): Descriptor for selecting fields from datasets.
        network (Module): Neural network model.
        loaders (tuple): Tuple of (train_loader, val_loader, test_loader).
        device (torch.device): Device to run the model on (e.g., torch.device("cuda")).
    """
    train_loader, _, test_loader = loaders
    train_data = get_all_data(train_loader.dataset)
    test_data = get_all_data(test_loader.dataset)

    fig, ax_pred = plt.subplots(figsize=(8, 6))
    ax_signal = ax_pred.twinx()

    # Ensure prediction axis overlays signal axis
    ax_pred.set_zorder(ax_signal.get_zorder() + 1)
    ax_pred.patch.set_visible(False)

    network.eval()

    # Extract identifiers
    run_ids = descriptor.select("run_id", train_data)
    unique_ids = torch.unique(run_ids).cpu().numpy()
    run_ids_np = run_ids.cpu().numpy().flatten()

    cmap = plt.get_cmap("tab10")

    # Plot training signals (faint background)
    for i, waveform_id in enumerate(unique_ids):
        mask = run_ids_np == waveform_id
        time = descriptor.select("time", train_data)[mask].cpu().numpy()
        signal = descriptor.select("signal", train_data)[mask].cpu().numpy().flatten()

        ax_signal.scatter(
            time,
            signal,
            color=cmap(i % 10),
            alpha=0.2,
            s=5,
            label=f"Training signal {int(waveform_id)}",
        )

    # Prepare test waveform
    time_test = descriptor.select("time", test_data).cpu().numpy()
    signal_test = descriptor.select("signal", test_data)
    signal_test_np = signal_test.cpu().numpy().flatten()

    # Forward pass
    with torch.no_grad():
        preds = network({"input": signal_test.to(device)})["output"].cpu().numpy().flatten()

    # Plot test signal and prediction
    ax_signal.scatter(
        time_test,
        signal_test_np,
        color="red",
        alpha=0.4,
        edgecolors="k",
        s=10,
        label="Test signal",
    )

    ax_pred.plot(
        time_test,
        preds,
        color="red",
        linewidth=2,
        alpha=0.8,
        label="Test prediction",
    )

    # Titles and labels
    ax_pred.set_title("Model Predictions and Signal")
    ax_pred.set_xlabel("x")
    ax_pred.set_ylabel("Prediction", color="red")
    ax_signal.set_ylabel("Signal", color="gray")
    ax_pred.set_ylim(-0.1, 1.1)

    ax_pred.tick_params(axis="y", labelcolor="red")
    ax_signal.tick_params(axis="y", labelcolor="gray")

    # Merge legends and reorder so test entries appear together at the end
    lines_pred, labels_pred = ax_pred.get_legend_handles_labels()
    lines_sig, labels_sig = ax_signal.get_legend_handles_labels()

    # Combine
    lines = lines_pred + lines_sig
    labels = labels_pred + labels_sig

    # Sort so test set entries appear last
    sorted_pairs = sorted(
        zip(lines, labels, strict=False), key=lambda x: ("test" not in x[1].lower(), x[1].lower())
    )
    lines, labels = zip(*sorted_pairs, strict=False)

    ax_pred.legend(lines, labels, loc="best")

    fig.tight_layout()
    return ax_pred
