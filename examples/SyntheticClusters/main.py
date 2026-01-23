import os
from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from plot import plot_decision_boundary
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from util import MLPNetworkWithSoftmax, NNLLossFromProb

from congrads.callbacks.base import Callback, CallbackManager
from congrads.callbacks.registry import LoggerCallback
from congrads.constraints.base import Constraint
from congrads.constraints.registry import (
    ImplicationConstraint,
    ScalarConstraint,
)
from congrads.core.congradscore import CongradsCore
from congrads.datasets.registry import SyntheticClusters
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.utils.utility import (
    CSVLogger,
    Seeder,
    split_data_loaders,
)


def main():
    # Argument parser
    parser = ArgumentParser(description="Run script with specified epochs.")
    parser.add_argument("--n_epoch", type=int, default=350, help="Number of epochs")
    args = parser.parse_args()

    # Set seed for reproducibility
    seeder = Seeder(base_seed=42)
    seeder.set_reproducible()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load and preprocess data
    dataset = SyntheticClusters(
        cluster_centers=[
            (0.3, 0.70),
            (0.25, 0.25),
            (0.13, 0.45),
            (0.35, 0.5),
            (0.67, 0.6),
            (0.80, 0.55),
            (0.75, 0.35),
            (0.55, 0.15),
            (0.6, 0.85),
        ],
        cluster_sizes=[100, 50, 20, 50, 50, 15, 100, 50, 50],
        cluster_std=[0.06, 0.07, 0.04, 0.06, 0.07, 0.04, 0.07, 0.06, 0.06],
        cluster_labels=[0, 0, 1, 0, 1, 0, 1, 0, 0],
    )
    loaders = split_data_loaders(
        dataset,
        loader_args={"batch_size": 100, "shuffle": True},
        valid_loader_args={"shuffle": False},
        test_loader_args={"shuffle": False},
    )

    # Instantiate network and push to correct device
    network = MLPNetworkWithSoftmax(n_inputs=2, n_outputs=2, n_hidden_layers=3, hidden_dim=50)
    network = network.to(device)

    # Instantiate loss and optimizer
    criterion = NNLLossFromProb()
    optimizer = Adam(network.parameters(), lr=0.001)

    # Descriptor setup
    descriptor = Descriptor()
    descriptor.add("input", "x", 0, constant=True)
    descriptor.add("input", "y", 1, constant=True)
    descriptor.add("output", "ProbA", 0)
    descriptor.add("output", "ProbB", 1)

    # Constraints definition
    Constraint.descriptor = descriptor
    Constraint.device = device
    constraints = [
        ImplicationConstraint(
            head=ScalarConstraint("x", "<=", 0.25),
            body=ScalarConstraint("ProbA", ">=", 0.7),
        ),
    ]

    # Initialize metric manager
    metric_manager = MetricManager()

    # Initialize data loggers
    tensorboard_logger = SummaryWriter(log_dir="logs/SyntheticClusters")
    csv_logger = CSVLogger("logs/SyntheticClusters.csv")
    logger_callback = LoggerCallback(
        metric_manager=metric_manager, tensorboard_logger=tensorboard_logger, csv_logger=csv_logger
    )

    # Callbacks setup
    plotting_callback = PlottingCallback(network, dataset)
    callback_manager = CallbackManager().add(plotting_callback).add(logger_callback)

    # Instantiate core
    core = CongradsCore(
        descriptor=descriptor,
        constraints=constraints,
        dataloader_train=loaders[0],
        dataloader_valid=loaders[1],
        dataloader_test=loaders[2],
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        metric_manager=metric_manager,
        callback_manager=callback_manager,
        device=device,
        enforce_all=True,
    )

    # Start/resume training
    core.fit(max_epochs=args.n_epoch)

    # Close writer
    tensorboard_logger.close()


class PlottingCallback(Callback):
    def __init__(self, network, dataset):
        super().__init__()
        self.network = network
        self.dataset = dataset

        os.makedirs("plots", exist_ok=True)

    def on_epoch_end(self, *args):
        plot_decision_boundary(self.network, self.dataset)
        plt.savefig("plots/SyntheticClusters.png")
        plt.close()


if __name__ == "__main__":
    main()
