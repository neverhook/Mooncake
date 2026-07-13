#!/usr/bin/env python3

"""Strict parsers and counter checks for the GB200 validation harness."""

from __future__ import annotations

import re
from dataclasses import dataclass


_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>-?\d+)\s*$"
)
_LABEL_RE = re.compile(
    r"(?:^|,)\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)="
    r'"(?P<value>(?:\\.|[^"\\])*)"\s*'
)


@dataclass(frozen=True)
class MetricSample:
    labels: dict[str, str]
    value: int


@dataclass(frozen=True)
class ProviderCapacity:
    requested_bytes: int
    effective_bytes: int
    chunk_count: int


@dataclass(frozen=True)
class ConsumerMetrics:
    cache_hits: int
    cache_misses: int
    lazy_imports: int
    lazy_import_duration_us: int


def _parse_labels(value: str | None) -> dict[str, str]:
    if value is None or not value.strip():
        return {}
    labels: dict[str, str] = {}
    cursor = 0
    while cursor < len(value):
        match = _LABEL_RE.match(value, cursor)
        if match is None:
            raise ValueError(f"invalid Prometheus labels: {value!r}")
        label_value = bytes(match.group("value"), "utf-8").decode("unicode_escape")
        labels[match.group("name")] = label_value
        cursor = match.end()
    return labels


def metric_samples(metrics: str, name: str) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for raw_line in metrics.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.fullmatch(line)
        if match is None or match.group("name") != name:
            continue
        samples.append(
            MetricSample(
                labels=_parse_labels(match.group("labels")),
                value=int(match.group("value")),
            )
        )
    return samples


def require_metric_value(
    metrics: str, name: str, labels: dict[str, str] | None = None
) -> int:
    expected_labels = labels or {}
    matches = [
        sample
        for sample in metric_samples(metrics, name)
        if all(
            sample.labels.get(key) == value for key, value in expected_labels.items()
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {name} sample with labels "
            f"{expected_labels!r}, found {len(matches)}"
        )
    return matches[0].value


def provider_capacity(metrics: str) -> ProviderCapacity:
    requested = require_metric_value(
        metrics, "mooncake_nvlink_host_numa_requested_capacity_bytes"
    )
    effective = require_metric_value(
        metrics, "mooncake_nvlink_host_numa_effective_capacity_bytes"
    )
    node_bytes = metric_samples(
        metrics, "mooncake_nvlink_host_numa_node_effective_bytes"
    )
    node_chunks = metric_samples(metrics, "mooncake_nvlink_host_numa_node_chunks")
    if not node_bytes or not node_chunks:
        raise ValueError("HOST_NUMA per-node capacity/chunk metrics are missing")
    if sum(sample.value for sample in node_bytes) != effective:
        raise ValueError(
            "HOST_NUMA per-node capacity does not sum to effective capacity"
        )
    chunk_count = sum(sample.value for sample in node_chunks)
    if chunk_count <= 0:
        raise ValueError("HOST_NUMA expected chunk count is not positive")
    return ProviderCapacity(requested, effective, chunk_count)


def consumer_metrics(metrics: str) -> ConsumerMetrics:
    return ConsumerMetrics(
        cache_hits=require_metric_value(
            metrics,
            "mooncake_nvlink_consumer_mapping_cache_total",
            {"result": "hit"},
        ),
        cache_misses=require_metric_value(
            metrics,
            "mooncake_nvlink_consumer_mapping_cache_total",
            {"result": "miss"},
        ),
        lazy_imports=require_metric_value(
            metrics, "mooncake_nvlink_consumer_lazy_import_observations_total"
        ),
        lazy_import_duration_us=require_metric_value(
            metrics, "mooncake_nvlink_consumer_lazy_import_duration_us_total"
        ),
    )


def consumer_delta(before: ConsumerMetrics, after: ConsumerMetrics) -> ConsumerMetrics:
    fields = ConsumerMetrics.__dataclass_fields__
    values: dict[str, int] = {}
    for field in fields:
        delta = getattr(after, field) - getattr(before, field)
        if delta < 0:
            raise ValueError(f"NVLink Consumer counter {field} decreased")
        values[field] = delta
    return ConsumerMetrics(**values)


def validate_miss_then_hit(
    first_transfer: ConsumerMetrics, subsequent_transfer: ConsumerMetrics
) -> None:
    if first_transfer.cache_misses <= 0:
        raise ValueError("first NVLink transfer did not record a mapping miss")
    if first_transfer.lazy_imports != first_transfer.cache_misses:
        raise ValueError(
            "first NVLink transfer mapping misses do not match lazy imports"
        )
    if first_transfer.lazy_import_duration_us <= 0:
        raise ValueError("first NVLink transfer did not record import latency")
    if subsequent_transfer.cache_hits <= 0:
        raise ValueError("subsequent NVLink transfer did not record a cache hit")
    if subsequent_transfer.cache_misses != 0 or subsequent_transfer.lazy_imports != 0:
        raise ValueError("subsequent NVLink transfer unexpectedly imported a mapping")


def consumer_delta_dict(delta: ConsumerMetrics) -> dict[str, int]:
    return {
        "hit": delta.cache_hits,
        "miss": delta.cache_misses,
        "lazy_imports": delta.lazy_imports,
        "lazy_import_duration_us": delta.lazy_import_duration_us,
    }


def classify_cache_phase(*deltas: ConsumerMetrics) -> str:
    hits = sum(delta.cache_hits for delta in deltas)
    misses = sum(delta.cache_misses for delta in deltas)
    if misses > 0 and hits > 0:
        return "mixed"
    if misses > 0:
        return "cold"
    if hits > 0:
        return "warm"
    return "none"
