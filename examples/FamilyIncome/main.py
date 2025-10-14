from argparse import ArgumentParser

import torch
from torch import ge, le
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from congrads.constraints import Constraint, ScalarConstraint, SumConstraint
from congrads.core import CongradsCore
from congrads.datasets import FamilyIncome
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.networks import MLPNetwork
from congrads.transformations import DenormalizeMinMax
from congrads.utils import (
    CSVLogger,
    Seeder,
    preprocess_FamilyIncome,
    split_data_loaders,
)

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser(description="Run script with specified epochs.")
    parser.add_argument("--n_epoch", type=int, default=80, help="Number of epochs")
    args = parser.parse_args()

    # Set seed for reproducibility
    seeder = Seeder(base_seed=42)
    seeder.set_reproducible()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load and preprocess data
    data = FamilyIncome("./datasets", preprocess_FamilyIncome, download=True)
    loaders = split_data_loaders(
        data,
        loader_args={"batch_size": 100, "shuffle": True},
        valid_loader_args={"shuffle": False},
        test_loader_args={"shuffle": False},
    )

    # Instantiate network and push to correct device
    network = MLPNetwork(n_inputs=24, n_outputs=8, n_hidden_layers=6, hidden_dim=80)
    network = network.to(device)

    # Instantiate loss and optimizer
    criterion = MSELoss()
    optimizer = Adam(network.parameters(), lr=0.0001)

    # Descriptor setup
    descriptor = Descriptor()
    descriptor.add("input", "Total Household Income", 0, constant=True)
    descriptor.add("output", "Total Food Expenditure", 0)
    descriptor.add("output", "Bread and Cereals Expenditure", 1)
    descriptor.add("output", "Meat Expenditure", 2)
    descriptor.add("output", "Vegetables Expenditure", 3)
    descriptor.add("output", "Housing and water Expenditure", 4)
    descriptor.add("output", "Medical Care Expenditure", 5)
    descriptor.add("output", "Communication Expenditure", 6)
    descriptor.add("output", "Education Expenditure", 7)

    # Constraints definition
    Constraint.descriptor = descriptor
    Constraint.device = device
    constraints = [
        ScalarConstraint(
            "Total Food Expenditure",
            ge,
            0,
        ),
        ScalarConstraint("Total Food Expenditure", le, 1),
        ScalarConstraint("Bread and Cereals Expenditure", ge, 0),
        ScalarConstraint("Bread and Cereals Expenditure", le, 1),
        ScalarConstraint("Meat Expenditure", ge, 0),
        ScalarConstraint("Meat Expenditure", le, 1),
        ScalarConstraint("Vegetables Expenditure", ge, 0),
        ScalarConstraint("Vegetables Expenditure", le, 1),
        ScalarConstraint("Housing and water Expenditure", ge, 0),
        ScalarConstraint("Housing and water Expenditure", le, 1),
        ScalarConstraint("Medical Care Expenditure", ge, 0),
        ScalarConstraint("Medical Care Expenditure", le, 1),
        ScalarConstraint("Communication Expenditure", ge, 0),
        ScalarConstraint("Communication Expenditure", le, 1),
        ScalarConstraint("Education Expenditure", ge, 0),
        ScalarConstraint("Education Expenditure", le, 1),
        SumConstraint(
            [
                DenormalizeMinMax("Total Food Expenditure", min=3704, max=791848),
            ],
            ge,
            [
                DenormalizeMinMax("Bread and Cereals Expenditure", min=0, max=437467),
                DenormalizeMinMax("Meat Expenditure", min=0, max=140992),
                DenormalizeMinMax("Vegetables Expenditure", min=0, max=74800),
            ],
        ),
        SumConstraint(
            [
                DenormalizeMinMax("Total Household Income", min=11285, max=11815988),
            ],
            ge,
            [
                DenormalizeMinMax("Total Food Expenditure", min=3704, max=791848),
                DenormalizeMinMax("Housing and water Expenditure", min=1950, max=2188560),
                DenormalizeMinMax("Medical Care Expenditure", min=0, max=1049275),
                DenormalizeMinMax("Communication Expenditure", min=0, max=149940),
                DenormalizeMinMax("Education Expenditure", min=0, max=731000),
            ],
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
    )

    # Initialize data loggers
    tensorboard_logger = SummaryWriter(log_dir="logs/FamilyIncome")
    csv_logger = CSVLogger("logs/FamilyIncome.csv")

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
        if epoch > 50 and epoch % 5:
            for g in optimizer.param_groups:
                g["lr"] /= 1.5

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

    # Start training
    core.fit(
        start_epoch=0,
        max_epochs=args.n_epoch,
        on_epoch_end=[on_epoch_end],
        on_test_end=[on_test_end],
    )

    # Close writer
    tensorboard_logger.close()
