import os
from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from plot import plot_regression_epoch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from congrads.callbacks.base import Callback, CallbackManager
from congrads.callbacks.registry import LoggerCallback
from congrads.constraints.base import Constraint
from congrads.constraints.registry import (
    ANDConstraint,
    ImplicationConstraint,
    RankedMonotonicityConstraint,
    ScalarConstraint,
)
from congrads.core.congradscore import CongradsCore
from congrads.datasets.registry import SyntheticMonotonicity
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.networks.registry import MLPNetwork
from congrads.utils.utility import (
    CSVLogger,
    Seeder,
    split_data_loaders,
)


def main():
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
    descriptor.add_layer("input", constant=True)
    descriptor.add_layer("output")
    descriptor.add_tag("x", "input", 0)
    descriptor.add_tag("y", "output", 0)

    # Constraints definition
    Constraint.descriptor = descriptor
    Constraint.device = device
    constraints = [
        ImplicationConstraint(
            head=ANDConstraint(
                ScalarConstraint("x", ">", 0.3),
                ScalarConstraint("x", "<=", 0.6),
            ),
            body=RankedMonotonicityConstraint("y", "x"),
        )
    ]

    # Initialize metric manager
    metric_manager = MetricManager()

    # Initialize data loggers
    tensorboard_logger = SummaryWriter(log_dir="logs/SimpleMonotonicity")
    csv_logger = CSVLogger("logs/SimpleMonotonicity.csv")
    logger_callback = LoggerCallback(
        metric_manager=metric_manager, tensorboard_logger=tensorboard_logger, csv_logger=csv_logger
    )

    # Callbacks setup
    plotting_callback = PlottingCallback(network, dataset, device)
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
    def __init__(self, network, dataset, device):
        super().__init__()
        self.network = network
        self.dataset = dataset
        self.device = device

        os.makedirs("plots", exist_ok=True)

    def on_epoch_end(self, *args):
        plot_regression_epoch(self.network, self.dataset, self.device)
        plt.savefig("plots/SimpleMonotonicity.png")
        plt.close()


if __name__ == "__main__":
    main()
