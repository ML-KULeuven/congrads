import warnings

import pytest

from congrads.callbacks.base import (
    Callback,
    CallbackManager,
    Operation,
)

# -----------------------
# Helper test operations
# -----------------------


class AddKey(Operation):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def compute(self, data, ctx):
        return {self.key: self.value}


class AddFromCtx(Operation):
    def __init__(self, key, ctx_key):
        self.key = key
        self.ctx_key = ctx_key

    def compute(self, data, ctx):
        return {self.key: ctx[self.ctx_key]}


class NoneReturningOp(Operation):
    def compute(self, data, ctx):
        return None


class InvalidReturningOp(Operation):
    def compute(self, data, ctx):
        return 123  # invalid


class ExplodingOp(Operation):
    def compute(self, data, ctx):
        raise ValueError("boom")


# -----------------------
# Operation tests
# -----------------------


def test_operation_returns_dict():
    op = AddKey("a", 1)
    out = op({}, {})
    assert out == {"a": 1}


def test_operation_none_is_empty_dict():
    op = NoneReturningOp()
    out = op({"x": 1}, {})
    assert out == {}


def test_operation_invalid_return_type():
    op = InvalidReturningOp()
    with pytest.raises(TypeError):
        op({}, {})


# -----------------------
# Callback tests
# -----------------------


class TestCallback(Callback):
    pass


def test_callback_executes_operations_in_order():
    cb = TestCallback()
    cb.add("on_batch_start", AddKey("a", 1))
    cb.add("on_batch_start", AddKey("b", 2))

    out = cb.on_batch_start({}, {})
    assert out == {"a": 1, "b": 2}


def test_callback_warns_on_key_collision():
    cb = TestCallback()
    cb.add("on_batch_start", AddKey("a", 1))
    cb.add("on_batch_start", AddKey("a", 2))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = cb.on_batch_start({}, {})

    assert out["a"] == 2
    assert any("overwriting keys" in str(warn.message) for warn in w)


def test_callback_wraps_operation_exception():
    cb = TestCallback()
    cb.add("on_batch_start", ExplodingOp())

    with pytest.raises(RuntimeError) as exc:
        cb.on_batch_start({}, {})

    assert "Error in operation" in str(exc.value)


# -----------------------
# CallbackManager tests
# -----------------------


def test_callback_manager_runs_callbacks_in_order():
    cb1 = TestCallback().add("on_batch_start", AddKey("a", 1))
    cb2 = TestCallback().add("on_batch_start", AddKey("b", 2))

    mgr = CallbackManager([cb1, cb2])
    out = mgr.run("on_batch_start", {})

    assert out == {"a": 1, "b": 2}


def test_callback_manager_shared_context():
    class SetCtx(Operation):
        def compute(self, data, ctx):
            ctx["x"] = 42
            return {}

    cb1 = TestCallback().add("on_batch_start", SetCtx())
    cb2 = TestCallback().add("on_batch_start", AddFromCtx("y", "x"))

    mgr = CallbackManager([cb1, cb2])
    out = mgr.run("on_batch_start", {})

    assert out == {"y": 42}
    assert mgr.ctx["x"] == 42


def test_callback_manager_empty_callback_is_noop():
    cb = TestCallback()
    mgr = CallbackManager([cb])

    data = {"x": 1}
    out = mgr.run("after_train_forward", data)

    # No ops → data flows through unchanged (but copied)
    assert out == data
    assert out is not data


def test_callback_manager_missing_stage_raises():
    class BadCallback:
        pass

    mgr = CallbackManager([BadCallback()])

    with pytest.raises(ValueError):
        mgr.run("on_batch_start", {})


def test_callback_manager_wraps_callback_exception():
    class BadCallback(Callback):
        def on_batch_start(self, data, ctx):
            raise RuntimeError("callback exploded")

    mgr = CallbackManager([BadCallback()])

    with pytest.raises(RuntimeError) as exc:
        mgr.run("on_batch_start", {})

    assert "Error in callback" in str(exc.value)


def test_callbacks_property_is_read_only():
    cb = TestCallback()
    mgr = CallbackManager([cb])

    callbacks = mgr.callbacks
    assert isinstance(callbacks, tuple)
    assert callbacks == (cb,)
