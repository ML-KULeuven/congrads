from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from plot import plot_decision_boundary
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from util import MLPNetworkWithSoftmax, NNLLossFromProb

from congrads.constraints import (
    Constraint,
    ImplicationConstraint,
    ScalarConstraint,
)
from congrads.core import CongradsCore
from congrads.datasets import SyntheticClusters
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.utils import (
    CSVLogger,
    Seeder,
    split_data_loaders,
)

if __name__ == "__main__":
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
            head=ScalarConstraint("x", torch.le, 0.25),
            body=ScalarConstraint("ProbA", torch.ge, 0.7),
        ),
    ]

    # Initialize metric manager
    metric_manager = MetricManager()

    # Instantiate core
    core = CongradsCore(
        descriptor,
        constraints,
        loaders,
        network,
        criterion,
        optimizer,
        metric_manager,
        device,
        enforce_all=False,
    )

    # Initialize data loggers
    tensorboard_logger = SummaryWriter(log_dir="logs/SyntheticClusters")
    csv_logger = CSVLogger("logs/SyntheticClusters.csv")

    def on_epoch_end(epoch: int):
        # Log metric values to TensorBoard and CSV file
        for name, value in metric_manager.aggregate("during_training").items():
            tensorboard_logger.add_scalar(name, value.item(), epoch)
            csv_logger.add_value(name, value.item(), epoch)

        # Write changes to disk
        tensorboard_logger.flush()
        csv_logger.save()

        # Reset metric manager
        metric_manager.reset("during_training")

        # Plotting
        plot_decision_boundary(network, dataset)
        plt.savefig("SyntheticClusters.png")
        plt.close()

    def on_test_end(epoch: int):
        # Log metric values to TensorBoard and CSV file
        for name, value in metric_manager.aggregate("after_training").items():
            tensorboard_logger.add_scalar(name, value.item(), epoch)
            csv_logger.add_value(name, value.item(), epoch)

        # Write changes to disk
        tensorboard_logger.flush()
        csv_logger.save()

        # Reset metric manager
        metric_manager.reset("after_training")

    # Start/resume training
    core.fit(max_epochs=args.n_epoch, on_epoch_end=[on_epoch_end], on_test_end=[on_test_end])

    # Close writer
    tensorboard_logger.close()
