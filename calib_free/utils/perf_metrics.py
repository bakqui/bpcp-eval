from typing import Dict, Tuple

import torch
import torchmetrics


def build_metric_fn(config: dict) -> Tuple[
    torchmetrics.Metric,
    Dict[str, float],
]:
    common_metric_fn_kwargs = {
        "compute_on_cpu": config["compute_on_cpu"],
        "sync_on_compute": config["sync_on_compute"],
    }

    metric_list = []
    for metric_class_name in config["target_metrics"]:
        if isinstance(metric_class_name, dict):
            # e.g., {"AUROC": {"average": macro}}
            assert len(metric_class_name) == 1, \
                f"Invalid metric name: {metric_class_name}"
            metric_class_name, metric_fn_kwargs = list(
                metric_class_name.items()
            )[0]
            metric_fn_kwargs.update(common_metric_fn_kwargs)
        else:
            metric_fn_kwargs = common_metric_fn_kwargs
        assert isinstance(metric_class_name, str), \
            f"metric name must be a string: {metric_class_name}"
        if hasattr(torchmetrics, metric_class_name):
            metric_class = getattr(torchmetrics, metric_class_name)
        else:
            raise ValueError(f"Invalid metric name: {metric_class_name}")
        metric_fn = metric_class(**metric_fn_kwargs)
        metric_list.append(metric_fn)
    metric_fn = torchmetrics.MetricCollection(metric_list)

    best_metrics = {
        k: -float("inf") if v.higher_is_better else float("inf")
        for k, v in metric_fn.items()
    }

    return metric_fn, best_metrics


def is_best_metric(
    metric_class: torchmetrics.Metric,
    prev_metric: float,
    curr_metric: float,
) -> bool:
    # check "higher_is_better" attribute of the metric class
    higher_is_better = metric_class.higher_is_better
    if higher_is_better:
        return curr_metric > prev_metric
    else:
        return curr_metric < prev_metric
