from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from plot import plot_regression_epoch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from util import split_dataset

from congrads.constraints import (
    BinaryConstraint,
    Constraint,
    GroupedMonotonicityConstraint,
    ImplicationConstraint,
    ScalarConstraint,
)
from congrads.core import CongradsCore
from congrads.datasets import SectionedGaussians
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.networks import MLPNetwork
from congrads.utils import (
    CSVLogger,
    Seeder,
)

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser(description="Run script with specified epochs.")
    parser.add_argument("--n_epoch", type=int, default=150, help="Number of epochs")
    args = parser.parse_args()

    # Set seed for reproducibility
    seeder = Seeder(base_seed=42)
    seeder.set_reproducible()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load and preprocess data
    sections = [
        {
            "range": (0.0, 0.40),
            "add_mean": 0.2,
            "std": 0.01,
            "max_splits": 1,
            "split_prob": 0.7,
            "mean_var": 0.6,
            "std_var": 0.0,
            "range_var": 0.9,
        },
        {
            "range": (0.40, 0.75),
            "add_mean": 1.8,
            "std": 0.01,
            "max_splits": 1,
            "split_prob": 0.4,
            "mean_var": 0.3,
            "std_var": 0.0,
            "range_var": 0.9,
        },
        {
            "range": (0.75, 1.0),
            "add_mean": 1.8,
            "std": 0.01,
            "max_splits": 0,
            "split_prob": 0.0,
            "mean_var": 0.4,
            "std_var": 0.0,
            "range_var": 0.9,
        },
    ]
    dataset = SectionedGaussians(
        sections,
        n_samples=600,
        n_runs=10,
        seed=seeder.roll_seed(),
        blend_k=50,
    )
    loaders = split_dataset(
        dataset,
        loader_args={"batch_size": 100, "shuffle": True},
        valid_loader_args={"shuffle": False},
        test_loader_args={"shuffle": False},
        seed=seeder.roll_seed(),
        train_valid_split=0.8,
    )

    # Instantiate network and push to correct device
    network = MLPNetwork(n_inputs=1, n_outputs=1, n_hidden_layers=4, hidden_dim=500)
    network = network.to(device)

    # Instantiate loss and optimizer
    criterion = MSELoss()  # ZeroLoss()
    optimizer = Adam(network.parameters(), lr=0.001)

    # Descriptor setup
    descriptor = Descriptor()
    descriptor.add("input", "signal", 0, constant=True)
    descriptor.add("context", "time", 0, constant=True)
    descriptor.add("context", "energy", 1, constant=True)
    descriptor.add("context", "run_id", 2, constant=True)
    descriptor.add("output", "score", 0)

    # Constraints definition
    Constraint.descriptor = descriptor
    Constraint.device = device
    constraints = [
        ScalarConstraint("score", torch.le, 1.05, rescale_factor=2.5),
        ScalarConstraint("score", torch.ge, -0.05, rescale_factor=2.5),
        ImplicationConstraint(
            head=ScalarConstraint("time", torch.le, 0.1),
            body=ScalarConstraint("score", torch.ge, 0.95, rescale_factor=2.0),
        ),
        ImplicationConstraint(
            head=ScalarConstraint("time", torch.ge, 0.9),
            body=ScalarConstraint("score", torch.le, 0.05, rescale_factor=2.0),
        ),
        GroupedMonotonicityConstraint(
            "score",
            "time",
            "run_id",
            direction="descending",
            rescale_factor_lower=1.50,
            rescale_factor_upper=1.75,
        ),
        ImplicationConstraint(
            head=ScalarConstraint("time", torch.le, 0.9),
            body=BinaryConstraint("score", torch.ge, "energy", rescale_factor=1.25),
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
    tensorboard_logger = SummaryWriter(log_dir="logs/MonotonicHealthScore")
    csv_logger = CSVLogger("logs/MonotonicHealthScore.csv")

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
        if epoch % 10 == 0 or epoch == args.n_epoch - 1:
            plot_regression_epoch(descriptor, network, loaders, device)
            plt.savefig("MonotonicHealthScore.png")
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
