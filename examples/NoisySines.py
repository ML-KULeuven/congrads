import random

import matplotlib.pyplot as plt
import torch
from torch import ge, gt, le, lt
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from congrads.checkpoints import CheckpointManager
from congrads.constraints import (
    Constraint,
    ImplicationConstraint,
    PythagoreanIdentityConstraint,
    ScalarConstraint,
)
from congrads.core import CongradsCore
from congrads.datasets import NoisySines
from congrads.descriptor import Descriptor
from congrads.metrics import MetricManager
from congrads.networks import MLPNetwork
from congrads.utils import CSVLogger, split_data_loaders

if __name__ == "__main__":

    # Set seed for reproducibility
    random.seed(42)
    seeds = []
    for i in range(4):
        seeds.append(random.randint(10, 10**6))
    torch.manual_seed(seeds[0])
    torch.cuda.manual_seed(seeds[1])
    torch.cuda.manual_seed_all(seeds[2])

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Load and preprocess data
    data = NoisySines(
        length=10000, noise_std=0.1, frequency=10, random_seed=seeds[3]
    )

    # Sample random points from data
    indices = list(range(len(data)))
    sampled_indices = random.sample(indices, 500)
    sampled_data = Subset(data, sampled_indices)

    loaders = split_data_loaders(
        sampled_data,
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
    network = MLPNetwork(1, 2, n_hidden_layers=4, hidden_dim=1024)
    network = network.to(device)

    # Instantiate loss and optimizer
    criterion = MSELoss()
    optimizer = Adam(network.parameters(), lr=0.0001)

    # Descriptor setup
    descriptor = Descriptor()
    descriptor.add("input", 0, "t", constant=True)
    descriptor.add("output", 0, "X")
    descriptor.add("output", 1, "Y")

    # Constraints definition
    Constraint.descriptor = descriptor
    Constraint.device = device
    constraints = [
        ScalarConstraint("X", le, 1.1, monitor_only=True),
        ScalarConstraint("X", ge, -1.1, monitor_only=True),
        ScalarConstraint("Y", le, 1.1, monitor_only=True),
        ScalarConstraint("Y", ge, -1.1, monitor_only=True),
        ImplicationConstraint(
            ScalarConstraint("t", lt, 0.5),
            ScalarConstraint("Y", ge, -0.1),
            monitor_only=True,
        ),
        ImplicationConstraint(
            ScalarConstraint("t", lt, 0.5),
            ScalarConstraint("Y", le, 0.1),
            monitor_only=True,
        ),
        ImplicationConstraint(
            ScalarConstraint("t", gt, 0.5),
            PythagoreanIdentityConstraint("X", "Y", atol=0.1, rtol=0),
            monitor_only=True,
        ),
    ]

    # Initialize metric manager
    metric_manager = MetricManager()

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        network,
        optimizer,
        metric_manager,
        save_dir="checkpoints/NoisyCausalSines",
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
    tensorboard_logger = SummaryWriter(log_dir="logs/NoisySines")
    csv_logger = CSVLogger("logs/NoisySines.csv")

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
        # if epoch % 5:
        #     for g in optimizer.param_groups:
        #         g["lr"] /= 2

        # Extract and convert data to tensors
        def extract_data(dataset):
            inputs, outputs = zip(*dataset)
            return torch.stack(inputs), torch.stack(outputs)

        train_inputs, train_outputs = extract_data(loaders[0].dataset)
        valid_inputs, valid_outputs = extract_data(loaders[1].dataset)
        test_inputs, test_outputs = extract_data(loaders[2].dataset)

        # Predict on validation set
        network.eval()
        valid_pred = network(valid_inputs.to(device))["output"].detach().cpu()
        test_pred = network(test_inputs.to(device))["output"].detach().cpu()

        # Plot training and validation data
        def plot_wave(index):
            plt.scatter(
                train_inputs.cpu(),
                train_outputs[:, index].cpu(),
                s=1,
                marker=".",
                c="orange",
                alpha=0.4,
                label="Training set",
            )
            plt.scatter(
                valid_inputs.cpu(),
                valid_outputs[:, index].cpu(),
                s=1,
                marker=".",
                c="orange",
                alpha=1,
                label="Validation set",
            )
            plt.scatter(
                test_inputs.cpu(),
                test_outputs[:, index].cpu(),
                s=1,
                marker=".",
                c="orange",
                alpha=0.6,
                label="Test set",
            )

            plt.scatter(
                valid_inputs,
                valid_pred[:, index],
                s=1,
                marker=".",
                c="blue",
                alpha=1,
                label="Validation predictions",
            )
            plt.scatter(
                test_inputs,
                test_pred[:, index],
                s=1,
                marker=".",
                c="blue",
                alpha=0.5,
                label="Test predictions",
            )

            plt.xlim(-0.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.gcf().suptitle(f"Epoch {epoch}")

        plt.clf()
        plt.subplot(2, 1, 1)
        plot_wave(0)
        plt.subplot(2, 1, 2)
        plot_wave(1)
        lgd = plt.legend(
            bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0
        )
        plt.savefig(
            "plots/foo.png", bbox_extra_artists=(lgd,), bbox_inches="tight"
        )

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

    # Start training
    start_epoch = checkpoint_manager.resume(ignore_missing=True)
    core.fit(
        start_epoch=start_epoch,
        max_epochs=1500,
        on_epoch_end=on_epoch_end,
        on_train_end=on_train_end,
    )

    # Close writer
    tensorboard_logger.close()
