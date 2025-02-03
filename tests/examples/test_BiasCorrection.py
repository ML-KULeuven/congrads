import random

import torch
from torch import ge, gt, le
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
    preprocess_BiasCorrection,
    split_data_loaders,
)


def test_BiasCorrection():

    # Set seed for reproducability
    random.seed(42)
    seeds = []
    for i in range(3):
        seeds.append(random.randint(10, 10**6))
    torch.manual_seed(seeds[0])
    torch.cuda.manual_seed(seeds[1])
    torch.cuda.manual_seed_all(seeds[2])

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Load and preprocess data
    data = BiasCorrection(
        "./datasets", preprocess_BiasCorrection, download=True
    )
    loaders = split_data_loaders(
        data,
        loader_args={
            "batch_size": 100,
            "shuffle": True,
            "num_workers": 6,
            "prefetch_factor": 2,
        },
        valid_loader_args={"shuffle": False},
        test_loader_args={"shuffle": False},
    )

    # Instantiate network and push to correct device
    network = MLPNetwork(25, 2, n_hidden_layers=3, hidden_dim=35)
    network = network.to(device)

    # Instantiate loss and optimizer
    criterion = MSELoss()
    optimizer = Adam(network.parameters(), lr=0.001)

    # Descriptor setup
    descriptor = Descriptor()
    descriptor.add("output", 0, "Tmax")
    descriptor.add("output", 1, "Tmin")

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
    checkpoint_manager = CheckpointManager(
        network,
        optimizer,
        metric_manager,
        save_dir="checkpoints/BiasCorrection",
        create_dir=True,
    )
    checkpoint_manager.register("CSR/valid")

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
        checkpoint_manager,
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

    def on_train_end(epoch: int):
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
        max_epochs=5,
        on_epoch_end=on_epoch_end,
        on_train_end=on_train_end,
    )

    # Close writer
    tensorboard_logger.close()
