import pytest
import torch

from congrads.metrics import Metric, MetricManager


# -------------------------------
# Tests for Metric
# -------------------------------
def test_metric_initialization():
    metric = Metric("loss")
    assert metric.name == "loss"
    assert metric.accumulator is not None
    assert metric.values == []
    assert metric.sample_count == 0


def test_metric_accumulate_and_sample_count():
    metric = Metric("loss")
    batch = torch.tensor([1.0, 2.0, 3.0])
    metric.accumulate(batch)
    assert len(metric.values) == 1
    assert metric.sample_count == 3
    # Check that the stored tensor is a clone, not the same object
    assert metric.values[0] is not batch


def test_metric_aggregate_single_batch():
    metric = Metric("loss")
    batch = torch.tensor([1.0, 2.0, 3.0])
    metric.accumulate(batch)
    aggregated = metric.aggregate()
    expected = torch.nanmean(batch)
    assert torch.allclose(aggregated, expected)


def test_metric_aggregate_multiple_batches():
    metric = Metric("loss")
    batch1 = torch.tensor([1.0, 2.0])
    batch2 = torch.tensor([3.0, 4.0])
    metric.accumulate(batch1)
    metric.accumulate(batch2)
    aggregated = metric.aggregate()
    expected = torch.nanmean(torch.cat([batch1, batch2]))
    assert torch.allclose(aggregated, expected)


def test_metric_aggregate_no_values_returns_nan():
    metric = Metric("loss")
    aggregated = metric.aggregate()
    assert torch.isnan(aggregated)


def test_metric_reset():
    metric = Metric("loss")
    metric.accumulate(torch.tensor([1.0]))
    metric.reset()
    assert metric.values == []
    assert metric.sample_count == 0


# -------------------------------
# Tests for MetricManager
# -------------------------------
@pytest.fixture
def metric_manager():
    return MetricManager()


def test_register_metric(metric_manager):
    metric_manager.register("loss")
    assert "loss" in metric_manager.metrics
    assert "loss" in metric_manager.groups
    assert metric_manager.groups["loss"] == "default"


def test_register_metric_custom_group_and_accumulator(metric_manager):
    metric_manager.register("accuracy", group="eval", accumulator=torch.mean)
    assert metric_manager.groups["accuracy"] == "eval"
    assert metric_manager.metrics["accuracy"].accumulator == torch.mean


def test_accumulate_and_aggregate_single_metric(metric_manager):
    metric_manager.register("loss")
    batch = torch.tensor([1.0, 2.0])
    metric_manager.accumulate("loss", batch)
    aggregated = metric_manager.aggregate()
    expected = torch.nanmean(batch)
    assert torch.allclose(aggregated["loss"], expected)


def test_accumulate_multiple_metrics_and_groups(metric_manager):
    metric_manager.register("loss", group="train")
    metric_manager.register("accuracy", group="eval")
    batch_loss = torch.tensor([1.0, 2.0])
    batch_acc = torch.tensor([0.8, 0.9])
    metric_manager.accumulate("loss", batch_loss)
    metric_manager.accumulate("accuracy", batch_acc)

    train_metrics = metric_manager.aggregate("train")
    eval_metrics = metric_manager.aggregate("eval")
    assert torch.allclose(train_metrics["loss"], torch.nanmean(batch_loss))
    assert torch.allclose(eval_metrics["accuracy"], torch.nanmean(batch_acc))


def test_reset_group(metric_manager):
    metric_manager.register("loss", group="train")
    metric_manager.register("accuracy", group="eval")
    metric_manager.accumulate("loss", torch.tensor([1.0]))
    metric_manager.accumulate("accuracy", torch.tensor([0.9]))

    metric_manager.reset("train")
    assert metric_manager.metrics["loss"].values == []
    # eval metric should remain unchanged
    assert metric_manager.metrics["accuracy"].values != []


def test_reset_all(metric_manager):
    metric_manager.register("loss", group="train")
    metric_manager.register("accuracy", group="eval")
    metric_manager.accumulate("loss", torch.tensor([1.0]))
    metric_manager.accumulate("accuracy", torch.tensor([0.9]))

    metric_manager.reset_all()
    assert metric_manager.metrics["loss"].values == []
    assert metric_manager.metrics["accuracy"].values == []


def test_accumulate_unregistered_metric_raises(metric_manager):
    with pytest.raises(KeyError, match="Metric 'unknown' is not registered."):
        metric_manager.accumulate("unknown", torch.tensor([1.0]))
