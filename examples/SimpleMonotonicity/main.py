from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from plot import plot_regression_epoch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from congrads.constraints import (
    ANDConstraint,
    Constraint,
    ImplicationConstraint,
    MonotonicityConstraint,
    ScalarConstraint,
)
from congrads.core import CongradsCore
from congrads.datasets import SyntheticMonotonicity
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.networks import MLPNetwork
from congrads.utils import (
    CSVLogger,
    Seeder,
    split_data_loaders,
)

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser(description="Run script with specified epochs.")
    parser.add_argument("--n_epoch", type=int, default=500, help="Number of epochs")
    args = parser.parse_args()

    # Set seed for reproducibility
    seeder = Seeder(base_seed=42)
    seeder.set_reproducible()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load and preprocess data
    dataset = SyntheticMonotonicity(
        n_samples=100,
        x_range=(0, 1),
        osc_amplitude=0.15,
        osc_frequency=20.0,
        osc_prob=1.0,
        noise_base=0.01,
        noise_scale=0.02,
        seed=seeder.roll_seed(),
    )
    loaders = split_data_loaders(
        dataset,
        loader_args={"batch_size": 10, "shuffle": True},
        valid_loader_args={"shuffle": False},
        test_loader_args={"shuffle": False},
    )

    # Instantiate network and push to correct device
    network = MLPNetwork(n_inputs=1, n_outputs=1, n_hidden_layers=3, hidden_dim=100)
    network = network.to(device)

    # Instantiate loss and optimizer
    criterion = MSELoss()
    optimizer = Adam(network.parameters(), lr=0.001)

    # Descriptor setup
    descriptor = Descriptor()
    descriptor.add("input", "x", 0, constant=True)
    descriptor.add("output", "y", 0)

    # Constraints definition
    Constraint.descriptor = descriptor
    Constraint.device = device
    constraints = [
        ImplicationConstraint(
            head=ANDConstraint(
                ScalarConstraint("x", torch.gt, 0.3),
                ScalarConstraint("x", torch.le, 0.6),
            ),
            body=MonotonicityConstraint("y", "x"),
        )
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
    tensorboard_logger = SummaryWriter(log_dir="logs/SimpleMonotonicity")
    csv_logger = CSVLogger("logs/SimpleMonotonicity.csv")

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
        plot_regression_epoch(network, dataset, device)
        plt.savefig("SimpleMonotonicity.png")
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
