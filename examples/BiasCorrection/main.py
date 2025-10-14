from argparse import ArgumentParser

import torch
from torch import Tensor, ge, gt, le
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from congrads.checkpoints import CheckpointManager
from congrads.constraints import BinaryConstraint, Constraint, ScalarConstraint
from congrads.core import CongradsCore
from congrads.datasets import BiasCorrection
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.networks import MLPNetwork
from congrads.utils import (
    CSVLogger,
    Seeder,
    preprocess_BiasCorrection,
    split_data_loaders,
)

if __name__ == "__main__":
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
        checkpoint_manager=checkpoint_manager,
    )

    # Initialize data loggers
    tensorboard_logger = SummaryWriter(log_dir="logs/BiasCorrection")
    csv_logger = CSVLogger("logs/BiasCorrection.csv")

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

        # Halve learning rate each 5 epochs
        if epoch % 5:
            for g in optimizer.param_groups:
                g["lr"] /= 2

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
    start_epoch = checkpoint_manager.resume(ignore_missing=True)
    core.fit(
        start_epoch=start_epoch,
        max_epochs=args.n_epoch,
        on_epoch_end=[on_epoch_end],
        on_test_end=[on_test_end],
    )

    # Close writer
    tensorboard_logger.close()
