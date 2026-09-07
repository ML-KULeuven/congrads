from argparse import ArgumentParser

import torch
from config import CGGDConfig
from data import Loader
from plot import PlotCallback
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from congrads.callbacks.base import CallbackManager
from congrads.callbacks.registry import LoggerCallback
from congrads.constraints.base import Constraint
from congrads.constraints.registry import (
    ImplicationConstraint,
    PDEConstraint,
    ScalarConstraint,
)
from congrads.core.congradscore import CongradsCore
from congrads.core.constraint_engine import ConstraintEngine
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
    parser.add_argument("--n_epoch", type=int, default=5000, help="Number of epochs")
    args = parser.parse_args()

    # Load configuration
    config = CGGDConfig()

    # Set seed for reproducibility
    seeder = Seeder(base_seed=42)
    seeder.set_reproducible()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load and preprocess data
    loader = Loader(config=config, base_seed=seeder.roll_seed())
    train_loader, valid_loader, test_loader = loader.load()

    # Instantiate network
    network = CustomNetwork(
        config=config, n_inputs=2, n_outputs=1, n_hidden_layers=3, hidden_dim=100
    )
    network = network.to(device)

    # Instantiate loss and optimizer
    criterion = BoundaryMSE()
    optimizer = Adam(network.parameters(), lr=1e-4)

    # Descriptor setup
    descriptor = Descriptor()
    descriptor.add_layer("input", constant=True)
    descriptor.add_layer("output")
    descriptor.add_layer("bc_residual", affects_loss=False, gradients_from="final_layer")
    descriptor.add_layer("ic_residual", affects_loss=False, gradients_from="final_layer")
    descriptor.add_layer("col_residual", affects_loss=False, gradients_from="final_layer")
    descriptor.add_layer("final_layer")

    descriptor.add_tag("x", "input", 0)
    descriptor.add_tag("t", "input", 1)
    descriptor.add_tag("bc_residual", "bc_residual", 0)
    descriptor.add_tag("ic_residual", "ic_residual", 0)
    descriptor.add_tag("col_residual", "col_residual", 0)
    descriptor.add_tag("final_layer", "final_layer", 0)

    # Constraints definition
    Constraint.descriptor = descriptor
    Constraint.device = device
    constraints = [
        ImplicationConstraint(
            head=ScalarConstraint("x", "<=", 0),
            body=PDEConstraint(
                layer_base="final_layer",
                tag_residual="bc_residual",
                comparator="<",
                scalar=0.001,
                rescale_factor=2.0,
            ),
        ),
        ImplicationConstraint(
            head=ScalarConstraint("x", ">=", 1),
            body=PDEConstraint(
                layer_base="final_layer",
                tag_residual="bc_residual",
                comparator="<",
                scalar=0.001,
                rescale_factor=2.0,
            ),
        ),
        ImplicationConstraint(
            head=ScalarConstraint("t", "<=", 0),
            body=PDEConstraint(
                layer_base="final_layer",
                tag_residual="ic_residual",
                comparator="<",
                scalar=0.001,
                rescale_factor=2.0,
            ),
        ),
        PDEConstraint(
            layer_base="final_layer",
            tag_residual="col_residual",
            comparator="<",
            scalar=0.001,
            rescale_factor=1.5,
        ),
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
    plotting_callback = PlotCallback(config, network, loader, device)
    callback_manager = CallbackManager().add(plotting_callback).add(logger_callback)

    # Instantiate core
    core = CongradsCore(
        descriptor=descriptor,
        constraints=constraints,
        dataloader_train=train_loader,
        dataloader_valid=valid_loader,
        dataloader_test=test_loader,
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        metric_manager=metric_manager,
        callback_manager=callback_manager,
        device=device,
        enforce_all=True,
        network_uses_grad=True,
        constraint_engine_cls=CustomConstraintEngine,
    )

    # Start/resume training
    core.fit(max_epochs=args.n_epoch)

    # Close writer
    tensorboard_logger.close()


class CustomNetwork(MLPNetwork):
    """Adjusts MLPNetwork to calculate several residuals using gradients."""

    def __init__(
        self,
        config: CGGDConfig,
        n_inputs,
        n_outputs,
        n_hidden_layers=4,
        hidden_dim=50,
        activation=torch.nn.Tanh(),
    ):
        self.config = config

        # Initialize MLPNetwork without final layer
        super().__init__(
            n_inputs=n_inputs,
            n_outputs=hidden_dim,
            n_hidden_layers=n_hidden_layers - 1,
            hidden_dim=hidden_dim,
            activation=activation,
        )

        # Add final layer manually
        self.final_linear = torch.nn.Linear(hidden_dim, n_outputs)
        self.final_activation = torch.nn.Tanh()

    def forward(self, data):
        # Enable gradient tracking for input
        data["input"].requires_grad_()

        # Get base network output without final layer
        data = super().forward(data)

        # Reconstruct output while keeping track of before last layer
        data["final_layer"] = self.final_activation(data["output"]).requires_grad_()
        data["output"] = self.final_linear(data["final_layer"])

        # Extract input and output tensors
        input = data["input"]
        output = data["output"]

        # Extract spatial and temporal coordinates
        x, t = input[:, 0].unsqueeze(1), input[:, 1].unsqueeze(1)

        # Compute 1st order gradients
        du_dX = torch.autograd.grad(output.sum(), input, retain_graph=True, create_graph=True)[0]
        du_dx = du_dX[:, 0].unsqueeze(1)
        du_dt = du_dX[:, 1].unsqueeze(1)

        # Compute 2nd order gradient
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), input, retain_graph=True, create_graph=True)[0]
        d2u_dx2 = d2u_dx2[:, 0].unsqueeze(1)

        # Compute residuals
        bc_residual = self.config.kappa * d2u_dx2
        ic_residual = (
            du_dt
            + self.config.kappa
            * torch.sin(torch.pi * x / self.config.L)
            * torch.pi**2
            / self.config.L**2
        )
        col_residual = self.config.kappa * d2u_dx2 - du_dt

        # Add residuals to data dict
        data["bc_residual"] = bc_residual.abs()
        data["ic_residual"] = ic_residual.abs()
        data["col_residual"] = col_residual.abs()

        return data


class BoundaryMSE(_Loss):
    """Computes MSE loss only for boundary samples (initial and boundary)."""

    def forward(self, output: Tensor, target: Tensor, data: dict[str, Tensor]) -> Tensor:
        # Mask for boundary samples
        is_boundary = (data["context"][:, 0] != 0).float().unsqueeze(1)

        # Per-sample MSE
        mse_per_sample = torch.mean(torch.square(output - target), dim=1, keepdim=True)

        # Avoid division by zero if no samples match
        denominator = is_boundary.sum().clamp_min(1.0)
        boundary_loss = torch.sum(mse_per_sample * is_boundary) / denominator

        return boundary_loss


class CustomConstraintEngine(ConstraintEngine):
    """Overrides loss gradients for collocation samples based on boundary sample gradients."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_buffer = None

    def _override_loss_gradients(self, norm_loss_grad, loss, data):
        # Mask to select only boundary samples (initial and boundary conditions)
        is_boundary = data["context"][:, 0] != 0

        if is_boundary.any():
            output_gradients = norm_loss_grad["final_layer"]

            # Mask gradients to boundary only and compute the maximum gradient
            masked_gradients = output_gradients[is_boundary]
            aggregated_gradient = masked_gradients.amax()

            # Set gradients of collocation samples to the maximum gradient of boundary samples
            norm_loss_grad["final_layer"][~is_boundary] = aggregated_gradient

            # Keep track of previous norm_loss_grad
            self.grad_buffer = aggregated_gradient

        else:
            # If no boundary samples in batch, fall back to aggregated gradient of previous batch
            norm_loss_grad["final_layer"][~is_boundary] = self.grad_buffer

        return norm_loss_grad


if __name__ == "__main__":
    main()
