from argparse import ArgumentParser

import torch
from torch import Tensor, ge, gt, le
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from congrads.callbacks.base import CallbackManager
from congrads.callbacks.registry import LoggerCallback
from congrads.checkpoints import CheckpointManager
from congrads.constraints.base import Constraint
from congrads.constraints.registry import BinaryConstraint, ScalarConstraint
from congrads.core.congradscore import CongradsCore
from congrads.datasets.registry import BiasCorrection
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.networks.registry import MLPNetwork
from congrads.utils.preprocessors import preprocess_BiasCorrection
from congrads.utils.utility import (
    CSVLogger,
    Seeder,
    split_data_loaders,
)


def main():
    # Argument parser
    parser = ArgumentParser(description="Run script with specified epochs.")
    parser.add_argument("--n_epoch", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()

    # Set seed for reproducibility
    seeder = Seeder(base_seed=42)
    seeder.set_reproducible()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load and preprocess data
    data = BiasCorrection("./datasets", preprocess_BiasCorrection, download=True)
    loaders = split_data_loaders(
        data,
        loader_args={"batch_size": 100, "shuffle": True},
        valid_loader_args={"shuffle": False},
        test_loader_args={"shuffle": False},
    )

    # Instantiate network and push to correct device
    network = MLPNetwork(n_inputs=25, n_outputs=2, n_hidden_layers=3, hidden_dim=35)
    network = network.to(device)

    # Instantiate loss and optimizer
    criterion = MSELoss()
    optimizer = Adam(network.parameters(), lr=0.001)

    # Descriptor setup
    descriptor = Descriptor()
    descriptor.add("output", "Tmax", 0)
    descriptor.add("output", "Tmin", 1)

    # Constraints definition
    Constraint.descriptor = descriptor
    Constraint.device = device
    constraints = [
        ScalarConstraint("Tmin", ge, 0),
        ScalarConstraint("Tmin", le, 1),
        ScalarConstraint("Tmax", ge, 0),
        ScalarConstraint("Tmax", le, 1),
        BinaryConstraint("Tmax", gt, "Tmin"),
    ]

    # Initialize metric manager
    metric_manager = MetricManager()

    # Initialize checkpoint manager
    def checkpointing_criterium(
        current_metrics_values: dict[str, Tensor],
        best_metric_values: dict[str, Tensor],
    ):
        current_csr_valid = current_metrics_values["CSR/valid"]
        best_csr_valid = best_metric_values.get("CSR/valid", 0)
        return current_csr_valid > best_csr_valid

    checkpoint_manager = CheckpointManager(
        checkpointing_criterium,
        network,
        optimizer,
        metric_manager,
        save_dir="checkpoints/BiasCorrection",
        create_dir=True,
        report_save=True,
    )

    # Initialize data loggers
    tensorboard_logger = SummaryWriter(log_dir="logs/BiasCorrection")
    csv_logger = CSVLogger("logs/BiasCorrection.csv")
    logger_callback = LoggerCallback(
        metric_manager=metric_manager, tensorboard_logger=tensorboard_logger, csv_logger=csv_logger
    )

    # Callbacks setup
    callback_manager = CallbackManager().add(logger_callback)

    # Instantiate core
    core = CongradsCore(
        network=network,
        descriptor=descriptor,
        constraints=constraints,
        dataloader_train=loaders[0],
        dataloader_valid=loaders[1],
        dataloader_test=loaders[2],
        criterion=criterion,
        optimizer=optimizer,
        callback_manager=callback_manager,
        metric_manager=metric_manager,
        device=device,
        checkpoint_manager=checkpoint_manager,
    )

    # Start/resume training
    start_epoch = checkpoint_manager.resume(ignore_missing=True)
    core.fit(
        start_epoch=start_epoch,
        max_epochs=args.n_epoch,
    )

    # Close writer
    tensorboard_logger.close()


if __name__ == "__main__":
    main()
