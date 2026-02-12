from argparse import ArgumentParser

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from congrads.callbacks.base import CallbackManager
from congrads.callbacks.registry import LoggerCallback
from congrads.constraints.base import Constraint
from congrads.constraints.registry import ScalarConstraint, SumConstraint
from congrads.core.congradscore import CongradsCore
from congrads.datasets.registry import FamilyIncome
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.networks.registry import MLPNetwork
from congrads.transformations.registry import DenormalizeMinMax
from congrads.utils.preprocessors import preprocess_FamilyIncome
from congrads.utils.utility import (
    CSVLogger,
    Seeder,
    split_data_loaders,
)


def main():
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
    descriptor.add_layer("input", constant=True)
    descriptor.add_layer("output")
    descriptor.add_tag("Total Household Income", "input", 0)
    descriptor.add_tag("Total Food Expenditure", "output", 0)
    descriptor.add_tag("Bread and Cereals Expenditure", "output", 1)
    descriptor.add_tag("Meat Expenditure", "output", 2)
    descriptor.add_tag("Vegetables Expenditure", "output", 3)
    descriptor.add_tag("Housing and water Expenditure", "output", 4)
    descriptor.add_tag("Medical Care Expenditure", "output", 5)
    descriptor.add_tag("Communication Expenditure", "output", 6)
    descriptor.add_tag("Education Expenditure", "output", 7)

    # Constraints definition
    Constraint.descriptor = descriptor
    Constraint.device = device
    constraints = [
        ScalarConstraint("Total Food Expenditure", ">=", 0),
        ScalarConstraint("Total Food Expenditure", "<=", 1),
        ScalarConstraint("Bread and Cereals Expenditure", ">=", 0),
        ScalarConstraint("Bread and Cereals Expenditure", "<=", 1),
        ScalarConstraint("Meat Expenditure", ">=", 0),
        ScalarConstraint("Meat Expenditure", "<=", 1),
        ScalarConstraint("Vegetables Expenditure", ">=", 0),
        ScalarConstraint("Vegetables Expenditure", "<=", 1),
        ScalarConstraint("Housing and water Expenditure", ">=", 0),
        ScalarConstraint("Housing and water Expenditure", "<=", 1),
        ScalarConstraint("Medical Care Expenditure", ">=", 0),
        ScalarConstraint("Medical Care Expenditure", "<=", 1),
        ScalarConstraint("Communication Expenditure", ">=", 0),
        ScalarConstraint("Communication Expenditure", "<=", 1),
        ScalarConstraint("Education Expenditure", ">=", 0),
        ScalarConstraint("Education Expenditure", "<=", 1),
        SumConstraint(
            [
                DenormalizeMinMax("Total Food Expenditure", min=3704, max=791848),
            ],
            ">=",
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
            ">=",
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

    # Initialize data loggers
    tensorboard_logger = SummaryWriter(log_dir="logs/FamilyIncome")
    csv_logger = CSVLogger("logs/FamilyIncome.csv")
    logger_callback = LoggerCallback(
        metric_manager=metric_manager, tensorboard_logger=tensorboard_logger, csv_logger=csv_logger
    )

    # Callbacks setup
    callback_manager = CallbackManager().add(logger_callback)

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
    )

    # Start training
    core.fit(
        start_epoch=0,
        max_epochs=args.n_epoch,
    )

    # Close writer
    tensorboard_logger.close()


if __name__ == "__main__":
    main()
