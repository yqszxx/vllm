# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 Flash attention on SM89.

Copied from the ROCm Triton path at upstream 658c8131c731, with the SM89
optimizations from backport fdd92607a3c6. Kept independent of ROCm dispatch.
"""

from dataclasses import dataclass, field
from typing import cast

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.ops.sm89_mla_sparse import (
    build_ragged_indices_from_dense,
    inv_rope_einsum,
    sparse_attn_decode,
    sparse_attn_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager

_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.int32).contiguous()
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    left_visible_ptr,
    right_visible_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SWA_WIDTH: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    PADDED_SWA_WIDTH: tl.constexpr,
    HAS_IMAGE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        if HAS_IMAGE:
            left = tl.load(left_visible_ptr + token_idx)
            right = tl.load(right_visible_ptr + token_idx)
        else:
            left = 0
            right = 0
        left_add = tl.maximum(left - (WINDOW_SIZE - 1), 0)
        # Prefix caching can resume inside an image span. Do not generate
        # indices outside the SWA rows present in the gathered workspace.
        swa_start = tl.maximum(
            tl.maximum(pos - (WINDOW_SIZE - 1) - left_add, 0), gather_start
        )
        swa_end = tl.minimum(pos + right + 1, seq_len)
        swa_len = tl.maximum(swa_end - swa_start, 0)

        topk_offset = tl.arange(0, PADDED_TOP_K)
        topk_mask = topk_offset < topk_len
        safe_topk_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + safe_topk_offset,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = (topk_indices >= 0) & (topk_indices < N)
        topk_indices = tl.where(valid_topk, topk_indices + M * batch_idx, -1)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offset,
            topk_indices,
            mask=topk_mask,
        )

        swa_offset = tl.arange(0, PADDED_SWA_WIDTH)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offset,
            M * batch_idx + N + swa_offset + swa_start - gather_start,
            mask=(swa_offset < swa_len) & (swa_offset < SWA_WIDTH),
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    max_image_tokens: int = 0,
    left_visible: torch.Tensor | None = None,
    right_visible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (left_visible is None) != (right_visible is None):
        raise ValueError("left_visible and right_visible must be provided together")
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    has_image = left_visible is not None
    # Keep the row shape fixed for a vision model even when a particular batch
    # has no image.
    swa_width = window_size + max_image_tokens
    combined_topk = (
        (topk + swa_width + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    num_workers = 128
    _combine_topk_swa_indices_kernel[(num_reqs, num_workers)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        left_visible if left_visible is not None else topk_indices,
        right_visible if right_visible is not None else topk_indices,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        SWA_WIDTH=swa_width,
        TOPK_WIDTH=topk_indices.shape[-1],
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
        PADDED_SWA_WIDTH=triton.next_power_of_2(swa_width),
        HAS_IMAGE=has_image,
    )
    return combined_indices, combined_lens


@triton.jit
def _compute_topk_lens_kernel(
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        count += tl.sum((local_idx >= 0).to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@triton.jit
def _pack_global_topk_ragged_kernel(
    global_topk_ragged_ptr,
    topk_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
    if block_idx * BLOCK_SIZE >= out_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    mask = (offset < out_len) & (offset < topk)
    local_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    valid = mask & (local_idx >= 0)
    block_indices = local_idx // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid,
        other=0,
    )
    block_offsets = local_idx % block_size
    slot_ids = tl.where(valid, block_numbers * block_size + block_offsets, -1)
    tl.store(global_topk_ragged_ptr + out_start + offset, slot_ids, mask=mask)


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


@dataclass
class _PrefillChunkSlices:
    """Prefill metadata slices for one request chunk.

    Cached on step metadata for the SM89 backend.
    """

    chunk_size: int
    query_start: int
    query_end: int
    seq_lens: torch.Tensor
    gather_lens: torch.Tensor
    swa_block_table: torch.Tensor
    query_start_loc: torch.Tensor
    compressed_seq_lens: torch.Tensor | None
    compressed_block_table: torch.Tensor | None


@dataclass
class DeepseekV4SM89Metadata(DeepseekV4FlashMLAMetadata):
    """DeepSeek V4 SM89 metadata carrying ragged decode topk."""

    c128a_decode_topk_ragged_indices: torch.Tensor | None = None
    c128a_decode_topk_ragged_indptr: torch.Tensor | None = None


@dataclass
class DeepseekV4SM89SWAMetadata(DeepseekSparseSWAMetadata):
    prefill_chunk_slices: dict[tuple, list[_PrefillChunkSlices]] = field(
        default_factory=dict
    )
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None


class DeepseekV4SM89MetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c128a_decode_topk_ragged_indices_buffer: torch.Tensor | None = None
        self.c128a_decode_topk_ragged_indptr_buffer: torch.Tensor | None = None
        if self.compress_ratio == 128:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            self.c128a_decode_topk_ragged_indices_buffer = torch.empty(
                max_tokens * self.c128a_max_compressed,
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_decode_topk_ragged_indptr_buffer = torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4SM89Metadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        dense_decode = base.c128a_global_decode_topk_indices
        decode_lens = base.c128a_decode_topk_lens
        if dense_decode is not None and decode_lens is not None:
            assert self.c128a_decode_topk_ragged_indices_buffer is not None
            assert self.c128a_decode_topk_ragged_indptr_buffer is not None
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                dense_decode.reshape(dense_decode.shape[0], -1),
                decode_lens,
                indices_out=self.c128a_decode_topk_ragged_indices_buffer[
                    : max(dense_decode.shape[0] * self.c128a_max_compressed, 1)
                ],
                indptr_out=self.c128a_decode_topk_ragged_indptr_buffer[
                    : dense_decode.shape[0] + 1
                ],
            )

        return DeepseekV4SM89Metadata(
            **vars(base),
            c128a_decode_topk_ragged_indices=ragged_indices,
            c128a_decode_topk_ragged_indptr=ragged_indptr,
        )


class DeepseekV4SM89SWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    # Speculative decoding is not supported by this backend.
    supports_draft_decode_metadata_update = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        swa_index_width = self.window_size
        self.decode_swa_ragged_indices_buffer = torch.empty(
            max_tokens * swa_index_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_ragged_indptr_buffer = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4SM89SWAMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        if (
            base.num_decode_tokens > 0
            and base.decode_swa_indices is not None
            and base.decode_swa_lens is not None
        ):
            assert self.decode_swa_ragged_indices_buffer is not None
            assert self.decode_swa_ragged_indptr_buffer is not None
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                base.decode_swa_indices.reshape(
                    base.num_decode_tokens, base.decode_swa_width
                ),
                base.decode_swa_lens,
                indices_out=self.decode_swa_ragged_indices_buffer[
                    : max(base.num_decode_tokens * base.decode_swa_width, 1)
                ],
                indptr_out=self.decode_swa_ragged_indptr_buffer[
                    : base.num_decode_tokens + 1
                ],
            )

        return DeepseekV4SM89SWAMetadata(
            **vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr,
        )


class DeepseekV4SM89Backend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4SM89MetadataBuilder"]:
        return DeepseekV4SM89MetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == (8, 9)


class DeepseekV4SM89SWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls():
        return DeepseekV4SM89SWAMetadataBuilder


class DeepseekV4SM89Attention(DeepseekV4Attention):
    _deepseek_v4_sm89 = True
    backend_cls = DeepseekV4SM89Backend
    swa_backend_cls = DeepseekV4SM89SWABackend

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Cached BF16 WO_A projection after inverse RoPE, followed by WO_B.
        z = inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        zf = z.flatten(1)
        return self.wo_b(zf)

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

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        sm89_metadata = cast(
            DeepseekV4SM89Metadata | None,
            attn_metadata.get(self.prefix),
        )
        swa_metadata = cast(
            DeepseekV4SM89SWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=sm89_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=sm89_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4SM89SWAMetadata,
        attn_metadata: DeepseekV4SM89Metadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                (
                    topk_ragged_indices,
                    topk_ragged_indptr,
                    topk_lens,
                ) = compute_global_topk_ragged_indices_and_indptr(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
                topk_ragged_indices = attn_metadata.c128a_decode_topk_ragged_indices
                topk_ragged_indptr = attn_metadata.c128a_decode_topk_ragged_indptr

        sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
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

    def _prefill_chunk_slices(
        self,
        attn_metadata: DeepseekV4SM89Metadata | None,
        swa_metadata: DeepseekV4SM89SWAMetadata,
    ) -> list[_PrefillChunkSlices]:
        cache_key = (
            self.compress_ratio,
            self.PREFILL_CHUNK_SIZE,
            None if attn_metadata is None else attn_metadata.block_table.data_ptr(),
        )
        cached = swa_metadata.prefill_chunk_slices.get(cache_key)
        if cached is not None:
            return cached

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None
        assert gather_lens is not None
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None

        num_decodes = swa_metadata.num_decodes
        prefill_token_base = int(query_start_loc_cpu[num_decodes])
        swa_block_table = swa_metadata.block_table[num_decodes:]
        compressed_block_table = (
            None if attn_metadata is None else attn_metadata.block_table[num_decodes:]
        )

        chunks: list[_PrefillChunkSlices] = []
        for chunk_start in range(0, swa_metadata.num_prefills, self.PREFILL_CHUNK_SIZE):
            chunk_end = min(
                chunk_start + self.PREFILL_CHUNK_SIZE, swa_metadata.num_prefills
            )
            chunk_seq_lens = seq_lens[chunk_start:chunk_end]
            chunks.append(
                _PrefillChunkSlices(
                    chunk_size=chunk_end - chunk_start,
                    query_start=int(query_start_loc_cpu[num_decodes + chunk_start])
                    - prefill_token_base,
                    query_end=int(query_start_loc_cpu[num_decodes + chunk_end])
                    - prefill_token_base,
                    seq_lens=chunk_seq_lens,
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    swa_block_table=swa_block_table[chunk_start:chunk_end],
                    query_start_loc=query_start_loc[
                        num_decodes + chunk_start : num_decodes + chunk_end + 1
                    ],
                    compressed_seq_lens=(
                        None
                        if compressed_block_table is None
                        else chunk_seq_lens // self.compress_ratio
                    ),
                    compressed_block_table=(
                        None
                        if compressed_block_table is None
                        else compressed_block_table[chunk_start:chunk_end]
                    ),
                )
            )

        swa_metadata.prefill_chunk_slices[cache_key] = chunks
        return chunks

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4SM89Metadata | None,
        swa_metadata: DeepseekV4SM89SWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        left_visible = swa_metadata.prefill_left_visible
        right_visible = swa_metadata.prefill_right_visible
        if left_visible is not None:
            left_visible = left_visible[num_decode_tokens:]
            assert right_visible is not None
            right_visible = right_visible[num_decode_tokens:]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk in self._prefill_chunk_slices(attn_metadata, swa_metadata):
            query_start, query_end = chunk.query_start, chunk.query_end
            chunk_size = chunk.chunk_size
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                assert chunk.compressed_seq_lens is not None
                assert chunk.compressed_block_table is not None
                # compressed_k_cache is OCP on every platform (Triton encoder).
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=chunk.compressed_seq_lens,
                    gather_lens=None,
                    block_table=chunk.compressed_block_table,
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )

            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=chunk.seq_lens,
                gather_lens=chunk.gather_lens,
                block_table=chunk.swa_block_table,
                block_size=swa_metadata.block_size,
                offset=N,
                use_fnuz=False,
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                chunk.query_start_loc,
                chunk.seq_lens,
                chunk.gather_lens,
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
                max_image_tokens=self.max_image_tokens,
                left_visible=(
                    left_visible[query_start:query_end]
                    if left_visible is not None
                    else None
                ),
                right_visible=(
                    right_visible[query_start:query_end]
                    if right_visible is not None
                    else None
                ),
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
