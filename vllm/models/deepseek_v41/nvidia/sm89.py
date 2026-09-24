# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 Flash attention on SM89.

Follows the V4.1 ROCm Triton flow without its AITER paths, on the SM89 copies
of the sparse attention kernels the DeepSeek V4 Flash path already uses.
Prefill gathers through the shared V4.1 chunk plan, as FlashMLA does.
"""

from typing import cast

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.models.deepseek_v4.nvidia.sm89 import (
    DeepseekV4SM89SWAMetadata,
    DeepseekV4SM89SWAMetadataBuilder,
    compute_global_topk_ragged_indices_and_indptr,
)
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    _replace_layer_index,
)
from vllm.models.deepseek_v41.common.ops import combine_topk_swa_indices
from vllm.models.deepseek_v41.common.ops.cache_utils import (
    dequantize_and_gather_k_cache_triton,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.attention.ops.sm89_mla_sparse import (
    inv_rope_einsum,
    sparse_attn_decode,
    sparse_attn_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

# (ragged_indices, ragged_indptr, lens) as consumed by sparse_attn_decode.
_TopkRagged = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class _LoadedWeight(QuantizeMethodBase):
    """Keeps a projection's checkpoint weight and scale exactly as loaded.

    The SM89 wo_a GEMM reads the one-byte weight and its E8M0 scales directly;
    the MXFP8 linear kernels would repack or widen them after loading.
    """

    supports_pre_processed_weights = True

    def create_weights(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass


class DeepseekV41SM89SWAMetadataBuilder(
    DeepseekV4SM89SWAMetadataBuilder, DeepseekV41SparseSWAMetadataBuilder
):
    pass


class DeepseekV41SM89Backend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == (8, 9)


class DeepseekV41SM89SWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekV41SM89SWAMetadataBuilder]:
        return DeepseekV41SM89SWAMetadataBuilder


class DeepseekV41SM89Attention(DeepseekV4Attention):
    _deepseek_v4_sm89 = True
    backend_cls = DeepseekV41SM89Backend
    swa_backend_cls = DeepseekV41SM89SWABackend
    # The dummy run reserves this many requests' worth of full-length
    # compressed context; the chunk plan still packs short requests together.
    PREFILL_CHUNK_SIZE = 1

    def __init__(self, *args, **kwargs):
        vllm_config: VllmConfig = args[0] if args else kwargs["vllm_config"]
        super().__init__(*args, **kwargs)
        if self.swa_cache_layer.bounded_replay:
            logger.warning_once(
                "SWA bounded replay is off for DeepSeek V4.1 Flash on SM89 (the "
                "Triton sparse prefill has no window clamp for replayed "
                "tokens); the sliding-window cache takes part in prefix "
                "caching instead."
            )
            self.swa_cache_layer.bounded_replay = False
        self.wo_a.quant_method = _LoadedWeight()
        self._topk_ragged_cache: dict[int, _TopkRagged] = {}
        self._index_source_prefix: str | None = None
        if self.compress_ratio > 0:
            assert self.index_source_layer_id is not None
            self._index_source_prefix = _replace_layer_index(
                self.prefix, self.index_source_layer_id
            )
            if self._index_source_prefix not in self._static_forward_context:
                raise NotImplementedError(
                    f"Index source {self._index_source_prefix} not found on "
                    "this rank; PP splits inside a v4.1 index-sharing group "
                    "are not supported."
                )

        if vllm_config.kernel_config.enable_jit_warmup:
            from vllm.models.deepseek_v41.common.ops.cache_utils import (
                _COMBINE_TOPK_SWA_INDICES_KERNEL,
            )

            _COMBINE_TOPK_SWA_INDICES_KERNEL.register_warmup()

        # Attention can be imported before VllmConfig enables breakable graphs.
        names = ["_sparse_indexer_and_attn"]
        if self._prepare_and_attn_fn == self._prepare_and_attn_eager:
            names.append("_prepare_and_attn_fn")
        for name in names:
            fn = getattr(self, name)
            if not hasattr(fn, "__wrapped__"):
                setattr(self, name, eager_break_during_capture(fn))

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        z = inv_rope_einsum(
            self.rotary_emb,
            attn_out[:, : self.n_local_heads, :],
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        return self._wo_b_proj(z.flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        attn_metadata = get_forward_context().attn_metadata
        swa_only = self.compress_ratio == 0

        if attn_metadata is None:
            # Warmup dummy run: reserve the workspace _forward_prefill takes.
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            top_k = 0
            if not swa_only:
                assert self.topk_indices_buffer is not None
                top_k = self.topk_indices_buffer.shape[-1]
            combined_topk = round_up(top_k + self.window_size, 128)
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix.
        compressed_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            DeepseekV4SM89SWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        compressed_k_cache = None if swa_only else self._compressed_kv_cache()
        num_decode_tokens = swa_metadata.num_decode_tokens

        if swa_metadata.num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=compressed_k_cache,
                output=output[num_decode_tokens:],
                attn_metadata=compressed_metadata,
                swa_metadata=swa_metadata,
            )
        if swa_metadata.num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=compressed_k_cache,
                swa_metadata=swa_metadata,
                attn_metadata=compressed_metadata,
                output=output[:num_decode_tokens],
            )

    def _decode_topk_ragged(
        self,
        swa_metadata: DeepseekV4SM89SWAMetadata,
        attn_metadata: DeepseekV4FlashMLAMetadata,
    ) -> _TopkRagged:
        """Ragged form of the topk indices this layer's index source published.

        It depends only on the shared ``topk_indices_buffer``, the step's
        metadata and the compress ratio, so the source memoizes one result per
        ratio for every layer that reads its indices.
        """
        assert swa_metadata.is_valid_token is not None
        assert self.topk_indices_buffer is not None
        assert self._index_source_prefix is not None

        source = self._static_forward_context[self._index_source_prefix]
        if source is self:
            # Fresh indices as of this layer; drop what the last step cached.
            source._topk_ragged_cache = {}
        cached = source._topk_ragged_cache.get(self.compress_ratio)
        if cached is not None:
            return cached

        num_decode_tokens = swa_metadata.num_decode_tokens
        built = compute_global_topk_ragged_indices_and_indptr(
            self.topk_indices_buffer[:num_decode_tokens],
            swa_metadata.token_to_req_indices,
            attn_metadata.block_table[: swa_metadata.num_decodes],
            attn_metadata.block_size // self.compress_ratio,
            swa_metadata.is_valid_token[:num_decode_tokens],
        )
        source._topk_ragged_cache[self.compress_ratio] = built
        return built

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4SM89SWAMetadata,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        output: torch.Tensor,
    ) -> None:
        swa_only = kv_cache is None
        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            assert attn_metadata is not None
            topk_ragged_indices, topk_ragged_indptr, topk_lens = (
                self._decode_topk_ragged(swa_metadata, attn_metadata)
            )

        sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=None,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekV4SM89SWAMetadata,
    ) -> None:
        swa_only = compressed_k_cache is None
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None
        assert gather_lens is not None
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        # Local indices filled by the index source; SWA-only layers pass
        # top_k=0 and never read them.
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:]
        topk_indices = topk_indices[: swa_metadata.num_prefill_tokens]
        top_k = 0 if swa_only else topk_indices.shape[-1]
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            has_compressed=not swa_only,
        )
        combined_topk = round_up(top_k + self.window_size, 128)
        swa_block_table = swa_metadata.block_table[num_decodes:]
        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv, combined_indices_out, combined_lens_out = (
                workspace_manager.get_simultaneous(
                    ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
                    ((self.max_num_batched_tokens, combined_topk), torch.int32),
                    ((self.max_num_batched_tokens,), torch.int32),
                )
            )
            if not swa_only:
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache_triton(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )
            dequantize_and_gather_k_cache_triton(
                kv[:chunk_size],
                self.swa_cache_layer.kv_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_N,
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            num_rows = query_end - query_start
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
                out=(combined_indices_out[:num_rows], combined_lens_out[:num_rows]),
            )
            sparse_attn_prefill(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices,
                topk_length=combined_lens,
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                output=output[query_start:query_end],
            )
