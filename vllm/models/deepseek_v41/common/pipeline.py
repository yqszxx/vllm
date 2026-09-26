# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-sharing dependencies of a DeepSeek V4.1 pipeline partition.

V4.1 layers read the compressed KV, indexer keys, top-k indices and candidate
blocks published by earlier source layers. A stage cut inside such a sharing
group leaves the source on an earlier stage than its consumers; the state then
has to travel with the activations, one hop per stage boundary it crosses.
"""

from dataclasses import dataclass
from typing import Any

# Cache-backed kinds travel as the rows written this step; the rest are the
# per-token contents of the model's shared top-k / candidate buffers.
CACHE_KINDS = ("kv", "index_k")


@dataclass(frozen=True)
class SharingDependency:
    kind: str
    source_layer: int
    consumer_layer: int
    source_stage: int
    consumer_stage: int


@dataclass(frozen=True, order=True)
class SharingRoute:
    """One hop of shared state from ``sender`` to ``sender + 1``."""

    kind: str
    source_layer: int
    sender: int

    @property
    def receiver(self) -> int:
        return self.sender + 1

    @property
    def key(self) -> str:
        return f"dsv41_pp.{self.kind}.{self.source_layer}"


def get_sharing_dependencies(
    config: Any, stage_ranges: list[tuple[int, int]]
) -> tuple[SharingDependency, ...]:
    """Resolve KV, index and candidate sources of every backbone layer."""
    num_layers = config.num_hidden_layers
    if (
        not stage_ranges
        or stage_ranges[0][0] != 0
        or stage_ranges[-1][1] != num_layers
        or any(start >= end for start, end in stage_ranges)
        or any(a[1] != b[0] for a, b in zip(stage_ranges, stage_ranges[1:]))
    ):
        raise ValueError("DeepSeek V4.1 pipeline stages must partition all layers")
    owners = {
        layer: stage
        for stage, (start, end) in enumerate(stage_ranges)
        for layer in range(start, end)
    }
    ratios = config.compress_ratios
    if len(ratios) < num_layers:
        raise ValueError("DeepSeek V4.1 compress_ratios must cover every layer")

    sources = {}
    for kind, field in (
        ("kv", "kv_source_layer_ids"),
        ("index", "index_source_layer_ids"),
    ):
        values = tuple(getattr(config, field, None) or ())
        if (
            any(
                type(layer) is not int or not 0 <= layer < num_layers
                for layer in values
            )
            or values != tuple(sorted(set(values)))
            or any(ratios[layer] == 0 for layer in values)
        ):
            raise ValueError(
                f"DeepSeek V4.1 {field} must contain sorted, unique compressed layers"
            )
        sources[kind] = values
    if not set(sources["kv"]).issubset(sources["index"]):
        raise ValueError("DeepSeek V4.1 KV sources must also publish index keys")

    dependencies = []

    def append(kind: str, source: int, consumer: int) -> None:
        if source != consumer:
            dependencies.append(
                SharingDependency(
                    kind, source, consumer, owners[source], owners[consumer]
                )
            )

    for layer in range(num_layers):
        if ratios[layer] == 0:
            continue
        for kind, values in sources.items():
            preceding = [source for source in values if source <= layer]
            if not preceding:
                raise ValueError(
                    f"DeepSeek V4.1 layer {layer} has no preceding {kind} source"
                )
            append(kind, preceding[-1], layer)
            # Non-owning index sources read the K cache of their kv source.
            if kind == "kv" and layer in sources["index"]:
                append("index_k", preceding[-1], layer)

    candidate = getattr(config, "candidate_source_layer_id", -1)
    if candidate >= 0 and getattr(config, "candidate_topk_blocks", 0) > 0:
        if candidate not in sources["index"]:
            raise ValueError("DeepSeek V4.1 candidate source must be an index source")
        for layer in sources["index"]:
            if layer > candidate:
                append("candidate", candidate, layer)
    return tuple(dependencies)


def get_sharing_routes(
    config: Any, dependencies: tuple[SharingDependency, ...]
) -> tuple[SharingRoute, ...]:
    """Hops needed by the cross-stage dependencies, relayed stage by stage.

    Only ratio-1 sources may cross a stage: their cache holds one row per
    token, so the rows written in a step are exactly that step's tokens. A
    ratio-2 source would also need its compressor's open-group state.
    """
    crossing = [d for d in dependencies if d.source_stage != d.consumer_stage]
    unsupported = [
        d
        for d in crossing
        if d.kind in CACHE_KINDS and config.compress_ratios[d.source_layer] != 1
    ]
    if unsupported:
        detail = "; ".join(
            f"{d.kind} source layer {d.source_layer} (stage {d.source_stage}, "
            f"compress_ratio {config.compress_ratios[d.source_layer]}) -> layer "
            f"{d.consumer_layer} (stage {d.consumer_stage})"
            for d in unsupported[:4]
        )
        raise NotImplementedError(
            "DeepSeek V4.1 pipeline partition splits a sharing group whose "
            f"source is not ratio 1: {detail}. Cut at a kv source layer instead."
        )
    return tuple(
        sorted(
            {
                SharingRoute(d.kind, d.source_layer, stage)
                for d in crossing
                for stage in range(d.source_stage, d.consumer_stage)
            }
        )
    )
