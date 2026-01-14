"""Manages the evaluation and optional enforcement of constraints on neural network outputs.

Responsibilities:
- Compute and log Constraint Satisfaction Rate (CSR) for training, validation, and test batches.
- Optionally adjust loss during training based on constraint directions and rescale factors.
- Handle gradient computation and CGGD application.
"""

import torch
from torch import Tensor, no_grad
from torch.linalg import vector_norm

from ..constraints.base import Constraint
from ..descriptor import Descriptor
from ..metrics import MetricManager


class ConstraintEngine:
    """Manages constraint evaluation and enforcement for a neural network.

    The ConstraintEngine coordinates constraints defined in Constraint objects,
    computes gradients for layers that affect the loss, logs metrics, and optionally
    modifies the loss during training according to the constraints. It supports
    separate phases for training, validation, and testing.
    """

    def __init__(
        self,
        *,
        constraints: list[Constraint],
        descriptor: Descriptor,
        metric_manager: MetricManager,
        device: torch.device,
        epsilon: float,
        aggregator: callable,
        enforce_all: bool,
    ) -> None:
        """Initialize the ConstraintEngine.

        Args:
            constraints: List of Constraint objects to evaluate and optionally enforce.
            descriptor: Descriptor containing metadata about network layers and which
                        variables affect the loss.
            metric_manager: MetricManager instance for logging CSR metrics.
            device: Torch device where tensors will be allocated (CPU or GPU).
            epsilon: Small positive value to avoid division by zero in gradient norms.
            aggregator: Callable used to reduce per-layer constraint contributions
                        to a scalar loss adjustment.
            enforce_all: Whether to enforce all constraints during training.
        """
        self.constraints = constraints
        self.descriptor = descriptor
        self.metric_manager = metric_manager
        self.device = device
        self.epsilon = epsilon
        self.enforce_all = enforce_all
        self.aggregator = aggregator

    def train(self, data: dict[str, Tensor], loss: Tensor) -> Tensor:
        """Apply all active constraints during training.

        Computes the original loss gradients for layers that affect the loss,
        evaluates each constraint, logs the Constraint Satisfaction Rate (CSR),
        and adjusts the loss according to constraint satisfaction.

        Args:
            data: Dictionary containing input and prediction tensors for the batch.
            loss: The original loss tensor computed from the network output.

        Returns:
            Tensor: The loss tensor after applying constraint-based adjustments.
        """
        return self._apply_constraints(data, loss, phase="train", enforce=True)

    def validate(self, data: dict[str, Tensor], loss: Tensor) -> Tensor:
        """Evaluate constraints during validation without modifying the loss.

        Computes and logs the Constraint Satisfaction Rate (CSR) for each constraint,
        but does not apply rescale adjustments to the loss.

        Args:
            data: Dictionary containing input and prediction tensors for the batch.
            loss: The original loss tensor computed from the network output.

        Returns:
            Tensor: The original loss tensor, unchanged.
        """
        return self._apply_constraints(data, loss, phase="valid", enforce=False)

    def test(self, data: dict[str, Tensor], loss: Tensor) -> Tensor:
        """Evaluate constraints during testing without modifying the loss.

        Computes and logs the Constraint Satisfaction Rate (CSR) for each constraint,
        but does not apply rescale adjustments to the loss.

        Args:
            data: Dictionary containing input and prediction tensors for the batch.
            loss: The original loss tensor computed from the network output.

        Returns:
            Tensor: The original loss tensor, unchanged.
        """
        return self._apply_constraints(data, loss, phase="test", enforce=False)

    def _apply_constraints(
        self, data: dict[str, Tensor], loss: Tensor, phase: str, enforce: bool
    ) -> Tensor:
        """Evaluate constraints, log CSR, and optionally adjust the loss.

        During training, computes loss gradients for variable layers that affect the loss.
        Iterates over all constraints, logging the Constraint Satisfaction Rate (CSR)
        and, if enforcement is enabled, adjusts the loss using constraint-specific
        directions and rescale factors.

        Args:
            data: Dictionary containing input and prediction tensors for the batch.
            loss: Original loss tensor computed from the network output.
            phase: Current phase, one of "train", "valid", or "test".
            enforce: If True, constraint-based adjustments are applied to the loss.

        Returns:
            Tensor: The combined loss after applying constraints (or the original loss
            if enforce is False or not in training phase).
        """
        total_rescale_loss = torch.tensor(0.0, device=self.device, dtype=loss.dtype)

        # Precompute gradients for variable layers affecting the loss
        if phase == "train":
            norm_loss_grad = {}
            variable_keys = self.descriptor.variable_keys & self.descriptor.affects_loss_keys

            for key in variable_keys:
                grad = torch.autograd.grad(
                    outputs=loss, inputs=data[key], retain_graph=True, allow_unused=True
                )[0]

                if grad is None:
                    raise RuntimeError(
                        f"Unable to compute loss gradients for layer '{key}'. "
                        "Set has_loss=False in Descriptor if this layer does not affect loss."
                    )

                grad_flat = grad.view(grad.shape[0], -1)
                norm_loss_grad[key] = (
                    vector_norm(grad_flat, dim=1, ord=2, keepdim=True)
                    .clamp(min=self.epsilon)
                    .detach()
                )

        # Iterate constraints
        for constraint in self.constraints:
            checks, mask = constraint.check_constraint(data)
            directions = constraint.calculate_direction(data)

            # Log CSR
            csr = (torch.sum(checks * mask) / torch.sum(mask)).unsqueeze(0)
            self.metric_manager.accumulate(f"{constraint.name}/{phase}", csr)
            self.metric_manager.accumulate(f"CSR/{phase}", csr)

            # Skip adjustment if not enforcing
            if not enforce or not constraint.enforce or not self.enforce_all or phase != "train":
                continue

            # Compute constraint-based rescale loss
            for key in constraint.layers & self.descriptor.variable_keys:
                with no_grad():
                    rescale = (1 - checks) * directions[key] * constraint.rescale_factor
                total_rescale_loss += self.aggregator(data[key] * rescale * norm_loss_grad[key])

        return loss + total_rescale_loss
