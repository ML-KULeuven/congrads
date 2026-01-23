import os
from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from plot import plot_regression_epoch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from util import split_dataset

from congrads.callbacks.base import Callback, CallbackManager
from congrads.callbacks.registry import LoggerCallback
from congrads.constraints.base import Constraint
from congrads.constraints.registry import (
    BinaryConstraint,
    ImplicationConstraint,
    PerGroupMonotonicityConstraint,
    RankedMonotonicityConstraint,
    ScalarConstraint,
)
from congrads.core.congradscore import CongradsCore
from congrads.datasets.registry import SectionedGaussians
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.networks.registry import MLPNetwork
from congrads.utils.utility import (
    CSVLogger,
    Seeder,
)


def main():
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
        ScalarConstraint("score", "<=", 1.05, rescale_factor=2.5),
        ScalarConstraint("score", ">=", -0.05, rescale_factor=2.5),
        ImplicationConstraint(
            head=ScalarConstraint("time", "<=", 0.1),
            body=ScalarConstraint("score", ">=", 0.95, rescale_factor=2.0),
        ),
        ImplicationConstraint(
            head=ScalarConstraint("time", ">=", 0.9),
            body=ScalarConstraint("score", "<=", 0.05, rescale_factor=2.0),
        ),
        PerGroupMonotonicityConstraint(
            base=RankedMonotonicityConstraint(
                "score",
                "time",
                direction="descending",
                rescale_factor_lower=1.50,
                rescale_factor_upper=1.75,
            ),
            tag_group="run_id",
        ),
        ImplicationConstraint(
            head=ScalarConstraint("time", "<=", 0.9),
            body=BinaryConstraint("score", ">=", "energy", rescale_factor=1.25),
        ),
    ]

    # Initialize metric manager
    metric_manager = MetricManager()

    # Initialize data loggers
    tensorboard_logger = SummaryWriter(log_dir="logs/MonotonicHealthScore")
    csv_logger = CSVLogger("logs/MonotonicHealthScore.csv")
    logger_callback = LoggerCallback(
        metric_manager=metric_manager, tensorboard_logger=tensorboard_logger, csv_logger=csv_logger
    )

    # Callbacks setup
    plotting_callback = PlottingCallback(descriptor, network, loaders, device)
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
    def __init__(self, descriptor, network, loaders, device):
        super().__init__()
        self.descriptor = descriptor
        self.network = network
        self.loaders = loaders
        self.device = device

        os.makedirs("plots", exist_ok=True)

    def on_epoch_end(self, data: dict, *args):
        epoch = data.get("epoch", 0)
        if epoch % 10 == 0:
            self._plot_figure()

    def on_train_end(self, *args):
        self._plot_figure()

    def _plot_figure(self):
        plot_regression_epoch(self.descriptor, self.network, self.loaders, self.device)
        plt.savefig("plots/MonotonicHealthScore.png")
        plt.close()


if __name__ == "__main__":
    main()
