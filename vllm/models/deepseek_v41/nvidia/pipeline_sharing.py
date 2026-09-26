# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Relay DeepSeek V4.1 shared state across a pipeline cut inside a sharing group.

A consumer stage keeps weight-free replicas of the remote source's compressed
KV and indexer K caches, registered under the source's layer names. The
global KV cache config therefore puts them in the source's groups, so their
block ids and slot mappings are the source's. Each step the sender gathers
the rows written at this step's slots, plus the shared top-k and candidate
buffers, into the intermediate tensors; the receiver writes them back before
its first layer. A stage between source and consumer does both.
"""

from typing import cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.distributed.utils import get_pp_indices
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4IndexerCache,
    _indexer_k_cache_head_dim,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4
from vllm.v1.kv_cache_interface import KVCacheSpec

from ..common.ops.pipeline_rows import gather_paged_rows, scatter_paged_rows
from ..common.pipeline import (
    CACHE_KINDS,
    SharingDependency,
    SharingRoute,
    get_sharing_dependencies,
    get_sharing_routes,
)

logger = init_logger(__name__)


class _AsKVSource:
    """A consumer layer viewed as the kv source whose cache it reads."""

    is_kv_source = True

    def __init__(self, layer: DeepseekV4Attention) -> None:
        self._layer = layer

    def __getattr__(self, name: str):
        return getattr(self._layer, name)


class DeepseekV41PipelineKVReplica(nn.Module, AttentionLayerBase):
    """Local copy of a remote kv source's compressed cache.

    Its spec and backend come from a local consumer, which shares the source's
    cache geometry, through the attention class's own code.
    """

    def __init__(
        self, vllm_config: VllmConfig, prefix: str, consumer_prefixes: list[str]
    ) -> None:
        super().__init__()
        assert consumer_prefixes
        self.prefix = prefix
        self.consumer_prefixes = consumer_prefixes
        self.kv_cache = torch.tensor([])
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self
        self._static_forward_context = context

    @property
    def donor(self) -> DeepseekV4Attention:
        return self._static_forward_context[self.consumer_prefixes[0]]

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        donor = self.donor
        return type(donor).get_kv_cache_spec(
            cast(DeepseekV4Attention, _AsKVSource(donor)), vllm_config
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.donor.get_attn_backend()

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        type(self.donor).bind_kv_cache(self, kv_cache)

    def forward(self):
        raise RuntimeError("Pipeline cache replicas do not execute attention")


class DeepseekV41PipelineSharing(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        stage: int,
        routes: tuple[SharingRoute, ...],
        dependencies: tuple[SharingDependency, ...],
        topk_indices_buffer: torch.Tensor,
        candidate_block_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.inbound = tuple(r for r in routes if r.receiver == stage)
        self.outbound = tuple(r for r in routes if r.sender == stage)
        self._prefix = prefix
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )
        self._topk_indices_buffer = topk_indices_buffer
        self._candidate_block_buffer = candidate_block_buffer
        self._max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._send_buffers: dict[str, torch.Tensor] = {}

        # Constructed before the decoder layers, which resolve their sources
        # through the forward context.
        self.replicas = nn.ModuleList()
        for route in self.inbound:
            source = route.source_layer
            if route.kind == "kv":
                consumers = sorted(
                    d.consumer_layer
                    for d in dependencies
                    if d.kind == "kv"
                    and d.source_layer == source
                    and d.consumer_stage == stage
                )
                assert all(
                    config.compress_ratios[layer] == config.compress_ratios[source]
                    for layer in consumers
                )
                self.replicas.append(
                    DeepseekV41PipelineKVReplica(
                        vllm_config,
                        self._cache_prefix(route),
                        [self._attn_prefix(layer) for layer in consumers],
                    )
                )
            elif route.kind == "index_k":
                self.replicas.append(
                    DeepseekV4IndexerCache(
                        head_dim=_indexer_k_cache_head_dim(
                            config.index_head_dim, dsa_indexer_uses_fp4(vllm_config)
                        ),
                        dtype=torch.uint8,
                        prefix=self._cache_prefix(route),
                        cache_config=vllm_config.cache_config,
                        compress_ratio=config.compress_ratios[source],
                    )
                )
            elif route.kind == "candidate":
                assert candidate_block_buffer is not None

        # Both read this step's slot mappings from the forward context.
        self._receive = eager_break_during_capture(self._receive_rows)
        self._send = eager_break_during_capture(self._send_rows)

    def _attn_prefix(self, layer: int) -> str:
        return f"{self._prefix}.layers.{layer}.attn"

    def _cache_prefix(self, route: SharingRoute) -> str:
        prefix = self._attn_prefix(route.source_layer)
        return f"{prefix}.indexer.k_cache" if route.kind == "index_k" else prefix

    def _token_buffer(self, route: SharingRoute) -> torch.Tensor:
        buffer = (
            self._topk_indices_buffer
            if route.kind == "index"
            else self._candidate_block_buffer
        )
        assert buffer is not None
        return buffer

    def _payload_layout(self, route: SharingRoute) -> tuple[int, torch.dtype]:
        if route.kind not in CACHE_KINDS:
            buffer = self._token_buffer(route)
            return buffer.shape[1], buffer.dtype
        layer = self._static_forward_context[self._cache_prefix(route)]
        if route.kind == "index_k":
            return layer.head_dim, torch.uint8
        if isinstance(layer, DeepseekV41PipelineKVReplica):
            layer = layer.donor
        if not layer.kv_cache_dtype.endswith("_ds_mla"):
            raise NotImplementedError(
                "DeepSeek V4.1 pipeline sharing relays packed KV records only, "
                f"not {layer.kv_cache_dtype}"
            )
        return layer.compressed_bytes_per_token, torch.uint8

    def make_empty_tensors(
        self, batch_size: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        tensors = {}
        for route in self.inbound:
            width, dtype = self._payload_layout(route)
            tensors[route.key] = torch.zeros(
                (batch_size, width), dtype=dtype, device=device
            )
        return tensors

    @staticmethod
    def _attn_metadata() -> dict | None:
        if not is_forward_context_available():
            return None
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, list):
            raise NotImplementedError(
                "DeepSeek V4.1 pipeline sharing does not support microbatching"
            )
        # Profile runs have no metadata and no bound caches.
        return attn_metadata if isinstance(attn_metadata, dict) else None

    def receive(self, intermediate_tensors) -> None:
        if self.inbound:
            self._receive(*(intermediate_tensors[r.key] for r in self.inbound))

    def _receive_rows(self, *payloads: torch.Tensor) -> None:
        attn_metadata = self._attn_metadata()
        if attn_metadata is None:
            return
        for route, values in zip(self.inbound, payloads):
            if route.kind in CACHE_KINDS:
                prefix = self._cache_prefix(route)
                scatter_paged_rows(
                    self._static_forward_context[prefix].kv_cache,
                    attn_metadata[prefix].slot_mapping,
                    values,
                )
            else:
                self._token_buffer(route)[: values.shape[0]].copy_(values)

    def send(self, num_tokens: int) -> dict[str, torch.Tensor]:
        if not self.outbound:
            return {}
        if not self._send_buffers:
            # The profile run allocates these, before any graph capture.
            assert not torch.cuda.is_current_stream_capturing()
            for route in self.outbound:
                width, dtype = self._payload_layout(route)
                self._send_buffers[route.key] = torch.zeros(
                    (self._max_num_tokens, width),
                    dtype=dtype,
                    device=self._topk_indices_buffer.device,
                )
        self._send(num_tokens)
        return {key: buffer[:num_tokens] for key, buffer in self._send_buffers.items()}

    def _send_rows(self, num_tokens: int) -> None:
        attn_metadata = self._attn_metadata()
        if attn_metadata is None:
            return
        for route in self.outbound:
            rows = self._send_buffers[route.key][:num_tokens]
            if route.kind in CACHE_KINDS:
                prefix = self._cache_prefix(route)
                gather_paged_rows(
                    self._static_forward_context[prefix].kv_cache,
                    attn_metadata[prefix].slot_mapping,
                    rows,
                )
            else:
                rows.copy_(self._token_buffer(route)[:num_tokens])


def maybe_build_pipeline_sharing(
    vllm_config: VllmConfig,
    prefix: str,
    topk_indices_buffer: torch.Tensor,
    candidate_block_buffer: torch.Tensor | None,
) -> DeepseekV41PipelineSharing | None:
    """Relay for this stage when the pipeline cut splits a sharing group."""
    pp_group = get_pp_group()
    if pp_group.world_size == 1:
        return None
    config = vllm_config.model_config.hf_config
    stage_ranges = [
        get_pp_indices(config.num_hidden_layers, rank, pp_group.world_size)
        for rank in range(pp_group.world_size)
    ]
    dependencies = get_sharing_dependencies(config, stage_ranges)
    routes = get_sharing_routes(config, dependencies)
    if not routes:
        return None
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.use_ubatching
        or parallel_config.decode_context_parallel_size != 1
        or parallel_config.prefill_context_parallel_size != 1
    ):
        raise NotImplementedError(
            "DeepSeek V4.1 pipeline cuts inside a sharing group do not support "
            "microbatching or context parallelism"
        )
    stage = pp_group.rank_in_group
    logger.info_once(
        "DeepSeek V4.1 pipeline stages %s split sharing groups; relaying %s",
        str(stage_ranges),
        ", ".join(f"{r.key} {r.sender}->{r.receiver}" for r in routes),
    )
    if not any(stage in (r.sender, r.receiver) for r in routes):
        return None
    return DeepseekV41PipelineSharing(
        vllm_config,
        prefix,
        stage,
        routes,
        dependencies,
        topk_indices_buffer,
        candidate_block_buffer,
    )
