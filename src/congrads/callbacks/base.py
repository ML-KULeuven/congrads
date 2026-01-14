"""Callback and Operation Framework for Modular Training Pipelines.

This module provides a structured system for defining and executing
callbacks and operations at different stages of a training lifecycle.
It is designed to support:

- Stateless, reusable operations that produce outputs merged into
  the event-local data.
- Callbacks that group operations and/or custom logic for specific
  stages of training, epochs, batches, and steps.
- A central CallbackManager to orchestrate multiple callbacks,
  maintain shared context, and execute stage-specific pipelines
  in deterministic order.

Stages supported:
    - on_train_start
    - on_train_end
    - on_epoch_start
    - on_epoch_end
    - on_batch_start
    - on_batch_end
    - on_test_start
    - on_test_end
    - on_train_batch_start
    - on_train_batch_end
    - on_valid_batch_start
    - on_valid_batch_end
    - on_test_batch_start
    - on_test_batch_end
    - after_train_forward
    - after_valid_forward
    - after_test_forward

Usage:
    1. Define Operations by subclassing `Operation` and implementing
       the `compute` method.
    2. Create a Callback subclass or instance and register Operations
       to stages via `add(stage, operation)`.
    3. Register callbacks with `CallbackManager`.
    4. Invoke `CallbackManager.run(stage, data)` at appropriate points
       in the training loop, passing in event-local data.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal, Self

Stage = Literal[
    "on_train_start",
    "on_train_end",
    "on_epoch_start",
    "on_epoch_end",
    "on_test_start",
    "on_test_end",
    "on_batch_start",
    "on_batch_end",
    "on_train_batch_start",
    "on_train_batch_end",
    "on_valid_batch_start",
    "on_valid_batch_end",
    "on_test_batch_start",
    "on_test_batch_end",
    "after_train_forward",
    "after_valid_forward",
    "after_test_forward",
]

STAGES: tuple[Stage, ...] = (
    "on_train_start",
    "on_train_end",
    "on_epoch_start",
    "on_epoch_end",
    "on_test_start",
    "on_test_end",
    "on_batch_start",
    "on_batch_end",
    "on_train_batch_start",
    "on_train_batch_end",
    "on_valid_batch_start",
    "on_valid_batch_end",
    "on_test_batch_start",
    "on_test_batch_end",
    "after_train_forward",
    "after_valid_forward",
    "after_test_forward",
)


class Operation(ABC):
    """Abstract base class representing a stateless unit of work executed inside a callback stage.

    Subclasses should implement the `compute` method which returns
    a dictionary of outputs to merge into the running event data.
    """

    def __repr__(self) -> str:
        """Return a concise string representation of the operation."""
        return f"<{self.__class__.__name__}>"

    def __call__(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute the operation with the given event-local data and shared context.

        Args:
            data (dict[str, Any]): Event-local dictionary containing data for this stage.
            ctx (dict[str, Any]): Shared context dictionary accessible by all operations and callbacks.

        Returns:
            dict[str, Any]: Outputs produced by the operation to merge into the running data.
                            Returns an empty dict if `compute` returns None.
        """
        out = self.compute(data, ctx)
        if out is None:
            return {}
        if not isinstance(out, dict):
            raise TypeError(f"{self.__class__.__name__}.compute must return dict or None")
        return out

    @abstractmethod
    def compute(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
        """Perform the operation's computation.

        Args:
            data (dict[str, Any]): Event-local dictionary containing the current data.
            ctx (dict[str, Any]): Shared context dictionary.

        Returns:
            dict[str, Any] or None: Outputs to merge into the running data.
                                     Returning None is equivalent to {}.
        """
        raise NotImplementedError


class Callback(ABC):  # noqa: B024
    """Abstract base class representing a callback that can have multiple operations registered to different stages of the training lifecycle.

    Each stage method executes all operations registered for that stage
    in insertion order. Operations can modify the event-local data dictionary.
    """

    def __init__(self):
        """Initialize the callback with empty operation lists for all stages."""
        self._ops_by_stage: dict[Stage, list[Operation]] = {s: [] for s in STAGES}

    def __repr__(self) -> str:
        """Return a concise string showing number of operations per stage."""
        ops_summary = {stage: len(ops) for stage, ops in self._ops_by_stage.items() if ops}
        return f"<{self.__class__.__name__} ops={ops_summary}>"

    def add(self, stage: Stage, op: Operation) -> Self:
        """Register an operation to execute at the given stage.

        Args:
            stage (Stage): Lifecycle stage at which to run the operation.
            op (Operation): Operation instance to add.

        Returns:
            Self: Returns self for method chaining.
        """
        self._ops_by_stage[stage].append(op)
        return self

    def _run_ops(self, stage: Stage, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute all operations registered for a specific stage.

        Args:
            stage (Stage): Lifecycle stage to execute.
            data (dict[str, Any]): Event-local data to pass to operations.
            ctx (dict[str, Any]): Shared context across callbacks and operations.

        Returns:
            dict[str, Any]: Merged data dictionary after executing all operations.

        Notes:
            - Operations are executed in insertion order.
            - If an operation overwrites existing keys, a warning is issued.
        """
        out = dict(data)

        for operation in self._ops_by_stage[stage]:
            try:
                produced = operation(out, ctx) or {}
            except Exception as e:
                raise RuntimeError(f"Error in operation {operation} at stage {stage}") from e

            collisions = set(produced.keys()) & set(out.keys())
            if collisions:
                import warnings

                warnings.warn(
                    f"Operation {operation} at stage '{stage}' is overwriting keys: {collisions}",
                    stacklevel=2,
                )

            out.update(produced)

        return out

    # --- training ---
    def on_train_start(self, data: dict[str, Any], ctx: dict[str, Any]):
        """Execute operations registered for the 'on_train_start' stage."""
        self._run_ops("on_train_start", data, ctx)

    def on_train_end(self, data: dict[str, Any], ctx: dict[str, Any]):
        """Execute operations registered for the 'on_train_end' stage."""
        self._run_ops("on_train_end", data, ctx)

    # --- epoch ---
    def on_epoch_start(self, data: dict[str, Any], ctx: dict[str, Any]):
        """Execute operations registered for the 'on_epoch_start' stage."""
        self._run_ops("on_epoch_start", data, ctx)

    def on_epoch_end(self, data: dict[str, Any], ctx: dict[str, Any]):
        """Execute operations registered for the 'on_epoch_end' stage."""
        self._run_ops("on_epoch_end", data, ctx)

    # --- test ---
    def on_test_start(self, data: dict[str, Any], ctx: dict[str, Any]):
        """Execute operations registered for the 'on_test_start' stage."""
        self._run_ops("on_test_start", data, ctx)

    def on_test_end(self, data: dict[str, Any], ctx: dict[str, Any]):
        """Execute operations registered for the 'on_test_end' stage."""
        self._run_ops("on_test_end", data, ctx)

    # --- batch ---
    def on_batch_start(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'on_batch_start' stage."""
        return self._run_ops("on_batch_start", data, ctx)

    def on_batch_end(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'on_batch_end' stage."""
        return self._run_ops("on_batch_end", data, ctx)

    def on_train_batch_start(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'on_train_batch_start' stage."""
        return self._run_ops("on_train_batch_start", data, ctx)

    def on_train_batch_end(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'on_train_batch_end' stage."""
        return self._run_ops("on_train_batch_end", data, ctx)

    def on_valid_batch_start(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'on_valid_batch_start' stage."""
        return self._run_ops("on_valid_batch_start", data, ctx)

    def on_valid_batch_end(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'on_valid_batch_end' stage."""
        return self._run_ops("on_valid_batch_end", data, ctx)

    def on_test_batch_start(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'on_test_batch_start' stage."""
        return self._run_ops("on_test_batch_start", data, ctx)

    def on_test_batch_end(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'on_test_batch_end' stage."""
        return self._run_ops("on_test_batch_end", data, ctx)

    def after_train_forward(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'after_train_forward' stage."""
        return self._run_ops("after_train_forward", data, ctx)

    def after_valid_forward(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'after_valid_forward' stage."""
        return self._run_ops("after_valid_forward", data, ctx)

    def after_test_forward(self, data: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute operations registered for the 'after_test_forward' stage."""
        return self._run_ops("after_test_forward", data, ctx)


class CallbackManager:
    """Orchestrates multiple callbacks and executes them at specific lifecycle stages.

    - Callbacks are executed in registration order.
    - Event-local data flows through all callbacks.
    - Shared context is available for cross-callback communication.
    """

    def __init__(self, callbacks: Iterable[Callback] | None = None):
        """Initialize a CallbackManager instance.

        Args:
        callbacks (Iterable[Callback] | None): Optional initial callbacks to register.
            If None, starts with an empty callback list.

        Attributes:
        _callbacks (list[Callback]): Internal list of registered callbacks.
        ctx (dict[str, Any]): Shared context dictionary accessible to all callbacks
            and operations for cross-event communication.
        """
        self._callbacks: list[Callback] = list(callbacks) if callbacks else []
        self.ctx: dict[str, Any] = {}

    def __repr__(self) -> str:
        """Return a concise representation showing registered callbacks and ctx keys."""
        names = [cb.__class__.__name__ for cb in self._callbacks]
        return f"<CallbackManager callbacks={names} ctx_keys={list(self.ctx.keys())}>"

    def add(self, callback: Callback) -> Self:
        """Register a single callback.

        Args:
            callback (Callback): Callback instance to add.

        Returns:
            Self: Returns self for fluent chaining.
        """
        self._callbacks.append(callback)
        return self

    def extend(self, callbacks: Iterable[Callback]) -> None:
        """Register multiple callbacks at once.

        Args:
            callbacks (Iterable[Callback]): Iterable of callbacks to add.
        """
        self._callbacks.extend(callbacks)

    def run(self, stage: Stage, data: dict[str, Any]) -> dict[str, Any]:
        """Execute all registered callbacks for a specific stage.

        Args:
            stage (Stage): Lifecycle stage to run (e.g., "on_batch_start").
            data (dict[str, Any]): Event-local data dictionary to pass through callbacks.

        Returns:
            dict[str, Any]: The final merged data dictionary after executing all callbacks.

        Raises:
            ValueError: If a callback does not implement the requested stage.
            RuntimeError: If any callback raises an exception during execution.
        """
        for cb in self._callbacks:
            if not hasattr(cb, stage):
                raise ValueError(
                    f"Callback {cb.__class__.__name__} has no handler for stage {stage}"
                )
            handler = getattr(cb, stage)

            try:
                new_data = handler(data, self.ctx)
                if new_data is not None:
                    data = new_data

            except Exception as e:
                raise RuntimeError(f"Error in callback {cb.__class__.__name__}.{stage}") from e

        return data

    @property
    def callbacks(self) -> tuple[Callback, ...]:
        """Return a read-only tuple of registered callbacks.

        Returns:
            tuple[Callback, ...]: Registered callbacks.
        """
        return tuple(self._callbacks)
