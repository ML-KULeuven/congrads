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

__all__ = ["ConstraintEngine"]


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

        self.norm_loss_grad: dict[str, Tensor] = {}

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

        if phase == "train":
            norm_loss_grad = self._calculate_loss_gradients(loss, data)
            norm_loss_grad = self._override_loss_gradients(norm_loss_grad, loss, data)

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
            for key in constraint.layers & self.descriptor.variable_layers:
                with no_grad():
                    rescale = (1 - checks) * directions[key] * constraint.rescale_factor

                # Determine which gradients to use based on the descriptor
                gradients_layer = self.descriptor.get_layer(key).gradients_from or key
                total_rescale_loss += self.aggregator(
                    data[key] * rescale * norm_loss_grad[gradients_layer]
                )

        return loss + total_rescale_loss

    def _calculate_loss_gradients(self, loss: Tensor, data: dict[str, Tensor]) -> None:
        """Compute and store normalized loss gradients for variable layers.

        For each layer that affects the loss, computes the gradient of the loss
        with respect to that layer's output. The gradients are normalized by their
        vector norms plus a small epsilon to avoid division by zero.

        Args:
            loss: The original loss tensor computed from the network output.
            data: Dictionary containing input and prediction tensors for the batch.
        """
        # Precompute gradients for variable layers affecting the loss
        norm_loss_grad = {}

        variable_keys = self.descriptor.variable_layers & self.descriptor.affects_loss_layers
        for key in variable_keys:
            if data[key].requires_grad is False:
                raise RuntimeError(
                    f"Layer '{key}' does not require gradients. Is this an input? "
                    "Set constant=True in Descriptor if this layer is an input."
                )

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
                vector_norm(grad_flat, dim=1, ord=2, keepdim=True).clamp(min=self.epsilon).detach()
            )

        return norm_loss_grad

    def _override_loss_gradients(
        self, norm_loss_grad: dict[str, Tensor], loss: Tensor, data: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Override the standard normalized loss gradient computation for custom functionality.

        Args:
            norm_loss_grad: Dictionary mapping parameter or component names to normalized
                gradient tensors.
            loss: Scalar loss value for the current training step.
            data: Dictionary containing the batch data used to compute the loss and any
                constraint-related signals.

        Returns:
            A dictionary of modified normalized gradients.
        """
        return norm_loss_grad
