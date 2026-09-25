# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 Flash attention on SM89.

Copied from the ROCm Triton path at upstream 658c8131c731, with the SM89
optimizations from backport fdd92607a3c6. Kept independent of ROCm dispatch.
"""

import functools
import math

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units

_DSV4_SPARSE_NOPE_DIM = 448
_DSV4_SPARSE_ROPE_DIM = 64


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    if scale.dtype == torch.uint8:
        # MXFP8 parameters preserve E8M0 scales as their raw exponent bytes.
        # They are biased exponents, not numeric uint8 scale values.
        return torch.exp2(scale.to(torch.int16).to(torch.float32) - 127.0)
    return scale.to(torch.float32)


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


@triton.jit
def _inverse_rope_gptj_kernel(
    o_ptr,  # [T, H, D] input
    out_ptr,  # [T, H, D] bf16 output
    pos_ptr,  # [T] positions
    cos_sin_ptr,  # [P, rope_dim] fp32 (cos[:half] | sin[half:])
    s_t,
    s_h,  # input row strides (last dim contiguous)
    os_t,
    os_h,  # output row strides
    cs_stride,  # cos_sin_cache row stride
    NOPE: tl.constexpr,  # non-rope head dims (passed through)
    HALF: tl.constexpr,  # rope_dim // 2
    BLOCK_NOPE: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    """Fused inverse GPT-J RoPE on the trailing rope_dim of each (token, head).

    Mirrors ``DeepseekV4ScalingRotaryEmbedding.forward_native(inverse=True)``
    for the GPT-J (non-neox) layout, writing bf16 directly. Replaces the
    clone + index_select + repeat_interleave + neg + stack + cat + cast chain
    (~10 small kernels) with a single launch.
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    in_base = t * s_t + h * s_h
    out_base = t * os_t + h * os_h

    # NoPE lanes pass through unchanged (only cast to bf16).
    n = tl.arange(0, BLOCK_NOPE)
    nmask = n < NOPE
    vals = tl.load(o_ptr + in_base + n, mask=nmask)
    tl.store(out_ptr + out_base + n, vals.to(tl.bfloat16), mask=nmask)

    # RoPE lanes: out_even = a*cos + b*sin, out_odd = b*cos - a*sin
    # (a = even lane, b = odd lane; sin negated for the inverse rotation).
    pos = tl.load(pos_ptr + t).to(tl.int64)
    k = tl.arange(0, BLOCK_HALF)
    kmask = k < HALF
    a = tl.load(o_ptr + in_base + NOPE + 2 * k, mask=kmask).to(tl.float32)
    b = tl.load(o_ptr + in_base + NOPE + 2 * k + 1, mask=kmask).to(tl.float32)
    cos = tl.load(cos_sin_ptr + pos * cs_stride + k, mask=kmask)
    sin = tl.load(cos_sin_ptr + pos * cs_stride + HALF + k, mask=kmask)
    out_even = a * cos + b * sin
    out_odd = b * cos - a * sin
    tl.store(out_ptr + out_base + NOPE + 2 * k, out_even.to(tl.bfloat16), mask=kmask)
    tl.store(out_ptr + out_base + NOPE + 2 * k + 1, out_odd.to(tl.bfloat16), mask=kmask)


def _fused_inverse_rope_gptj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    """bf16 inverse GPT-J RoPE via a single fused Triton kernel."""
    assert o.dim() == 3 and o.stride(-1) == 1, (
        "_fused_inverse_rope_gptj expects a [T, H, D] input with a contiguous last dim"
    )
    assert rope_head_dim > 0 and rope_head_dim % 2 == 0, (
        f"_fused_inverse_rope_gptj expects an even rope_head_dim, got {rope_head_dim}"
    )
    assert cos_sin_cache.shape[-1] == rope_head_dim, (
        "_fused_inverse_rope_gptj expects cos_sin_cache laid out as "
        f"[P, {rope_head_dim}] = cos | sin, got {tuple(cos_sin_cache.shape)}"
    )
    num_tokens, num_heads, head_dim = o.shape
    out = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=o.device
    )
    if num_tokens == 0:
        return out
    _inverse_rope_gptj_kernel[(num_tokens, num_heads)](
        o,
        out,
        positions,
        cos_sin_cache,
        o.stride(0),
        o.stride(1),
        out.stride(0),
        out.stride(1),
        cos_sin_cache.stride(0),
        NOPE=head_dim - rope_head_dim,
        HALF=rope_head_dim // 2,
        BLOCK_NOPE=triton.next_power_of_2(head_dim - rope_head_dim),
        BLOCK_HALF=triton.next_power_of_2(rope_head_dim // 2),
    )
    return out


def _wo_a_block_scaled(wo_a: torch.nn.Module) -> torch.Tensor | None:
    """The FP8 block scale of ``wo_a``, widened to fp32 and cached, or None if
    the weight is not raw block-scaled FP8.

    Emulated MXFP8 kernels can replace the one-byte weight with an already
    dequantized BF16 tensor while retaining the scale attribute for metadata;
    applying that retained scale again would double-dequantize the weight.
    """
    if wo_a.weight.element_size() != 1:
        return None
    cached = getattr(wo_a, "_dsv4_wo_a_scale", None)
    if cached is not None:
        return cached
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        get_fp8_block_weight_scale,
    )

    scale = get_fp8_block_weight_scale(wo_a)
    if scale is None:
        # ModelOpt MXFP8 stores the multiplicative E8M0 scale without the
        # historical ``_inv`` suffix.
        scale = getattr(wo_a, "weight_scale", None)
    if scale is None:
        return None
    cached = _decode_e8m0_scales(scale.data)
    wo_a._dsv4_wo_a_scale = cached
    return cached


def _wo_a_bmm(
    o_ref: torch.Tensor,
    wo_a: torch.nn.Module,
    n_local_groups: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """``o_ref`` [T, G, D] against the per-group wo_a weight [G, R, D].

    The GEMM applies the block scale inside the kernel, so no dequantized copy
    of the weight is ever resident -- a BF16 one costs 8.4 MB of KV cache per
    layer. Only prefill-sized batches, which can absorb it, dequantize.
    """
    from vllm.model_executor.kernels.linear.smallm_w8a16_sm89 import (
        can_implement,
        w8a16_block_scaled_smallm,
    )

    tokens, _, hidden_dim = o_ref.shape
    scale = _wo_a_block_scaled(wo_a)
    weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim)

    if scale is not None and can_implement(tokens, o_lora_rank, hidden_dim):
        scale = scale.view(n_local_groups, -1, scale.shape[-1])
        out = [
            w8a16_block_scaled_smallm(o_ref[:, g], weight[g], scale[g], o_ref.dtype)
            for g in range(n_local_groups)
        ]
        return out[0].unsqueeze(1) if n_local_groups == 1 else torch.stack(out, dim=1)

    if scale is not None:
        weight = weight.to(torch.float32) * _expand_2d_block_scales(
            scale.view(n_local_groups, -1, scale.shape[-1]), o_lora_rank, hidden_dim
        )
    return torch.einsum("tgd,grd->tgr", o_ref, weight.to(o_ref.dtype))


def inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """Inverse-RoPE + WO_A bmm path used on SM89.

    Fuses the inverse GPT-J RoPE into one Triton kernel and keeps wo_a in the
    FP8 form the checkpoint stores it in.
    """
    # ROCm folds this rotation into the sparse decode reduce (#57451, #57435).
    # Ported to SM89 it saved ~1-1.5 us per layer per step, well under 1% of a
    # decode step, while splitting the rotation between the decode epilogue
    # and a separate pass for prefill rows, so it is kept here for simplicity.
    o_ref = _fused_inverse_rope_gptj(
        o, positions, rotary_emb.cos_sin_cache, rope_head_dim
    )
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1)
    return _wo_a_bmm(o_ref, wo_a, n_local_groups, o_lora_rank)


def _validate_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    assert head_dim > 0, f"{op_name} expected a positive head_dim, got {head_dim}"
    assert nope_head_dim > 0, (
        f"{op_name} expected a positive NoPE dimension, got {nope_head_dim}"
    )
    assert rope_head_dim >= 0, (
        f"{op_name} expected a non-negative RoPE dimension, got {rope_head_dim}"
    )
    assert head_dim == nope_head_dim + rope_head_dim, (
        f"{op_name} expected head_dim={nope_head_dim + rope_head_dim}, got {head_dim}"
    )


def _validate_dsv4_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    _validate_sparse_dims(head_dim, nope_head_dim, rope_head_dim, op_name)
    assert (
        nope_head_dim == _DSV4_SPARSE_NOPE_DIM
        and rope_head_dim == _DSV4_SPARSE_ROPE_DIM
    ), (
        f"{op_name} expects {_DSV4_SPARSE_NOPE_DIM} NoPE dims and "
        f"{_DSV4_SPARSE_ROPE_DIM} RoPE dims"
    )


@triton.jit
def _pack_dense_prefix_to_ragged_kernel(
    indices_ptr,
    lengths_ptr,
    indptr_ptr,
    out_ptr,
    indices_stride0,
    num_rows_limit,
    row_width,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row_len = tl.load(lengths_ptr + row_idx)
    if block_idx * BLOCK_SIZE >= row_len:
        return

    mask = offsets < row_len
    safe_offsets = tl.where(offsets < row_width, offsets, 0)
    vals = tl.load(
        indices_ptr + row_idx * indices_stride0 + safe_offsets,
        mask=mask & (offsets < row_width),
        other=-1,
    ).to(tl.int32)
    if num_rows_limit >= 0:
        vals = tl.where((vals >= 0) & (vals < num_rows_limit), vals, -1)

    out_start = tl.load(indptr_ptr + row_idx)
    tl.store(out_ptr + out_start + offsets, vals, mask=mask)


@triton.jit
def _lengths_to_indptr_kernel(
    indptr_ptr,
    lengths_ptr,
    num_rows,
    BLOCK: tl.constexpr,
):
    """indptr[i] = sum(lengths[:i]) for i in [0, num_rows], in one launch."""
    offsets = tl.arange(0, BLOCK)
    vals = tl.load(lengths_ptr + offsets, mask=offsets < num_rows, other=0).to(tl.int32)
    tl.store(indptr_ptr + offsets + 1, tl.cumsum(vals, axis=0), mask=offsets < num_rows)
    tl.store(indptr_ptr + offsets, 0, mask=offsets == 0)


def _sm89_lengths_to_indptr(lengths: torch.Tensor, out: torch.Tensor) -> None:
    if lengths.numel() > 8191:
        out[:1].zero_()
        torch.cumsum(lengths, dim=0, out=out[1:])
        return
    _lengths_to_indptr_kernel[(1,)](
        out,
        lengths,
        lengths.numel(),
        BLOCK=triton.next_power_of_2(max(lengths.numel(), 1)),
        num_warps=8,
    )


def build_ragged_indices_from_dense(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    num_rows: int = -1,
    indices_out: torch.Tensor | None = None,
    indptr_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = indices.reshape(indices.shape[0], math.prod(indices.shape[1:]))
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    assert lengths.numel() == indices.shape[0], (
        f"Expected one length per row, got {lengths.shape} for indices {indices.shape}"
    )

    max_width = indices.shape[1] if indices.ndim == 2 else 0
    lengths = lengths.clamp(min=0, max=max_width).contiguous()

    if indptr_out is None:
        indptr = torch.zeros(
            indices.shape[0] + 1, dtype=torch.int32, device=indices.device
        )
        _sm89_lengths_to_indptr(lengths, indptr)
    else:
        assert indptr_out.dtype == torch.int32 and indptr_out.is_contiguous()
        assert indptr_out.shape == (indices.shape[0] + 1,)
        indptr = indptr_out
        _sm89_lengths_to_indptr(lengths, indptr)

    if indices.numel() == 0:
        flat = torch.empty(0, dtype=torch.int32, device=indices.device)
    else:
        if indices_out is None:
            flat = torch.empty(
                indices.shape[0] * max_width,
                dtype=torch.int32,
                device=indices.device,
            )
        else:
            assert indices_out.dtype == torch.int32 and indices_out.is_contiguous()
            assert indices_out.numel() >= indices.shape[0] * max_width
            flat = indices_out
        if flat.numel() > 0:
            block_size = 128
            _pack_dense_prefix_to_ragged_kernel[
                (indices.shape[0], triton.cdiv(max_width, block_size))
            ](
                indices,
                lengths,
                indptr,
                flat,
                indices.stride(0),
                int(num_rows),
                max_width,
                BLOCK_SIZE=block_size,
            )

    return flat, indptr


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


@triton.jit
def _sparse_kv_row_offset(slot, stride):
    # A global token slot fits in int32, but its byte/element offset may not.
    return slot.to(tl.int64) * stride


@triton.jit
def _sparse_attn_prefill_ragged_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    slot = tl.load(
        kv_indices_ptr + kv_start + k_offsets, mask=k_offsets < kv_len, other=-1
    )
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        valid = in_range & (slot >= 0) & (slot < num_kv)
        safe_slot = tl.where(valid, slot, 0)

        kv = tl.load(
            kv_ptr
            + _sparse_kv_row_offset(safe_slot[:, None], kv_stride_n)
            + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )

        next_k_pos = k_start + BLOCK_K + k_offsets
        slot = tl.load(
            kv_indices_ptr + kv_start + next_k_pos, mask=next_k_pos < kv_len, other=-1
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _decode_e8m0_scales_triton(encoded_scales):
    scale_bits = encoded_scales.to(tl.int32) << 23
    scale_bits = tl.where(encoded_scales == 0, 1 << 22, scale_bits)
    return scale_bits.to(tl.float32, bitcast=True)


@triton.jit
def _sparse_attn_decode_partial_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    q_stride0,
    q_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    # Both caches use OCP FP8 on SM89.
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    pid_h = tl.program_id(2)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    # Each split processes a contiguous slice of this query's main (SWA) and
    # extra (topk) segments. Slices are handled independently so a block never
    # straddles the main/extra boundary.
    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start
    main_chunk = (main_len + NUM_SPLITS - 1) // NUM_SPLITS
    main_lo = split_id * main_chunk
    main_hi = tl.minimum(main_lo + main_chunk, main_len)

    for k_start in tl.range(main_lo, main_hi, BLOCK_K, num_stages=NUM_STAGES):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_hi
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
        extra_chunk = (extra_len + NUM_SPLITS - 1) // NUM_SPLITS
        extra_lo = split_id * extra_chunk
        extra_hi = tl.minimum(extra_lo + extra_chunk, extra_len)

        for k_start in tl.range(extra_lo, extra_hi, BLOCK_K, num_stages=NUM_STAGES):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_hi
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope,
                tl.trans(k_rope),
            )
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    # Store raw (un-normalized) partial state for this split. Softmax sink and
    # final normalization happen in the reduce kernel.
    pm_base = query_idx * pm_stride0 + split_id * pm_stride_s + head_offsets
    tl.store(part_m_ptr + pm_base, m_i, mask=head_mask)
    tl.store(part_l_ptr + pm_base, l_i, mask=head_mask)
    acc_base = (
        part_acc_ptr
        + query_idx * pa_stride0
        + split_id * pa_stride_s
        + head_offsets[:, None] * pa_stride_h
    )
    tl.store(
        acc_base + nope_offsets[None, :],
        acc_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        acc_base + NOPE_DIM + rope_offsets[None, :],
        acc_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _sparse_attn_decode_reduce_kernel(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    attn_sink_ptr,
    out_ptr,
    out_stride0,
    out_stride1,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    COMB_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    comb_offsets = tl.arange(0, COMB_DIM)
    # SPLITS_PAD is NUM_SPLITS rounded up to a power of two so the parallel
    # split-axis load is a legal arange for any split count; padding lanes are
    # masked off.
    split_offsets = tl.arange(0, SPLITS_PAD)
    split_mask = split_offsets < NUM_SPLITS

    neg_large = -3.4028234663852886e38

    # Phase 1: load every split's running max/sum at once and reduce the max
    # in parallel (tl.max over the split axis) instead of walking the splits
    # serially. This breaks the long online-softmax dependency chain that made
    # the reduce latency-bound.
    load_mask = split_mask[:, None] & head_mask[None, :]
    pm_split = (
        part_m_ptr
        + query_idx * pm_stride0
        + split_offsets[:, None] * pm_stride_s
        + head_offsets[None, :]
    )
    m_all = tl.load(pm_split, mask=load_mask, other=neg_large)  # [S, H]
    l_all = tl.load(
        part_l_ptr
        + query_idx * pm_stride0
        + split_offsets[:, None] * pm_stride_s
        + head_offsets[None, :],
        mask=load_mask,
        other=0.0,
    )

    m_comb = tl.max(m_all, axis=0)  # [H]
    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_comb, sink)
    else:
        m_final = m_comb

    w_all = tl.exp(m_all - m_final[None, :])  # [S, H]
    w_all = tl.where(load_mask, w_all, 0.0)
    l_final = tl.sum(w_all * l_all, axis=0)  # [H]
    if HAS_ATTN_SINK:
        l_final = l_final + tl.exp(sink - m_final)
    denom = tl.maximum(l_final, 1.0e-30)

    # Phase 2: weighted sum of the per-split accumulators. The combine weight
    # for each split only depends on the (already known) global max, so the
    # acc loads carry no cross-split dependency and the compiler can pipeline
    # them; only the cheap FMA into `acc` is loop-carried.
    acc = tl.zeros((BLOCK_H, COMB_DIM), dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(
            part_m_ptr + query_idx * pm_stride0 + s * pm_stride_s + head_offsets,
            mask=head_mask,
            other=neg_large,
        )
        w_s = tl.exp(m_s - m_final)
        acc_base = (
            part_acc_ptr
            + query_idx * pa_stride0
            + s * pa_stride_s
            + head_offsets[:, None] * pa_stride_h
        )
        acc_s = tl.load(
            acc_base + comb_offsets[None, :],
            mask=head_mask[:, None],
            other=0.0,
        )
        acc += w_s[:, None] * acc_s

    out = tl.where(l_final[:, None] > 0.0, acc / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + comb_offsets[None, :],
        out,
        mask=head_mask[:, None],
    )


def _sparse_attn_prefill_ragged_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 1, f"expected indices=[nnz], got {indices.shape}"
    assert indptr.ndim == 1, f"expected indptr=[sq+1], got {indptr.shape}"
    assert not q.is_cpu and not kv.is_cpu and not indices.is_cpu and not indptr.is_cpu

    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    )
    _validate_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_sparse_attn_prefill_ragged_triton",
    )

    block_h = min(16, max(8, triton.next_power_of_2(num_heads)))
    block_d = triton.next_power_of_2(head_dim)
    block_k = 32
    num_warps = 4
    out = torch.empty_like(q)
    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out


def _sparse_attn_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
        indices,
        topk_length
        if topk_length is not None
        else (indices >= 0).sum(dim=-1, dtype=torch.int32),
        num_rows=kv.shape[0],
    )
    return _sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=ragged_indices,
        indptr=ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
    )


@functools.lru_cache
def _decode_cu_count() -> int:
    return num_compute_units()


def _decode_partial_iters(
    avg_main_len: float, avg_extra_len: float, splits: int, block_k: int
) -> int:
    """BLOCK_K iterations one partial workgroup walks for ``splits`` splits.

    Each split processes ``ceil(seg_len / splits)`` tokens of a segment, walked
    ``BLOCK_K`` at a time, and the main/extra segments are handled separately.
    """
    main_iters = (
        math.ceil(math.ceil(avg_main_len / splits) / block_k) if avg_main_len > 0 else 0
    )
    extra_iters = (
        math.ceil(math.ceil(avg_extra_len / splits) / block_k)
        if avg_extra_len > 0
        else 0
    )
    return main_iters + extra_iters


def _decode_num_splits(
    num_queries: int,
    heads_blocks: int,
    avg_main_len: float = 0.0,
    avg_extra_len: float = 0.0,
    block_k: int = 32,
) -> int:
    """Pick a flash-decode split count to keep the GPU busy across batch sizes.

    Decode launches only ``num_queries * heads_blocks`` workgroups otherwise,
    which severely under-fills the device for the low-concurrency regime that
    dominates latency. Splitting the KV sequence adds parallelism.

    We model the relative partial-kernel latency for a given split count ``s``
    as ``waves * (1/s + mu)`` where ``waves = ceil(base * s / CU)`` and ``mu``
    is a small per-wave overhead penalty:

      - ``waves / s`` captures the partial compute: each wave walks roughly
        ``total_tokens / s`` tokens and there are ``waves`` of them, so dividing
        by ``s`` makes more splits cheaper *until* they spill into extra waves.
      - ``mu * waves`` charges per-wave launch/tail overhead so we do not
        over-split into many mostly-idle waves (e.g. batch 224 on 256 CUs is
        best left at 1 split rather than 8 splits across 7 waves).

    The minimiser naturally prefers split counts that pack the device into full
    waves (``base * s`` near a multiple of ``CU``) and falls back to 1 split
    once the batch already fills the device. Ties favour the smaller split
    count (less reduce work).

    Finally we "snap down" the chosen split count to the smallest value that
    yields the same wave count *and* the same per-workgroup BLOCK_K iteration
    count. Because latency tracks iteration count (not raw token count), extra
    splits that do not lower the iteration count add only reduce/HBM overhead
    for no parallelism gain (e.g. batch 24: s8 and s10 both walk 4 extra iters
    in one wave, so s8 is strictly better). Snapping needs the average segment
    lengths, which the caller derives sync-free from the ragged index sizes.
    """
    base = max(1, num_queries * heads_blocks)
    # Target ~1 workgroup per CU: enough to fill the device while keeping the
    # reduce cost (which grows with split count) small. Tuned on gfx950.
    cu = max(1, _decode_cu_count())
    # Per-wave overhead penalty: higher values discourage split counts that
    # spill into extra GPU waves. Tuned on gfx950.
    mu = 0.04
    best_splits = 1
    best_cost = None
    # Search up to 16 splits; beyond that the reduce/HBM overhead dominates.
    for splits in range(1, 17):
        waves = (base * splits + cu - 1) // cu
        cost = waves * (1.0 / splits + mu)
        if best_cost is None or cost < best_cost - 1e-9:
            best_splits = splits
            best_cost = cost

    if best_splits > 1 and (avg_main_len > 0 or avg_extra_len > 0):
        target_waves = (base * best_splits + cu - 1) // cu
        target_iters = _decode_partial_iters(
            avg_main_len, avg_extra_len, best_splits, block_k
        )
        for splits in range(1, best_splits):
            waves = (base * splits + cu - 1) // cu
            iters = _decode_partial_iters(avg_main_len, avg_extra_len, splits, block_k)
            if waves == target_waves and iters == target_iters:
                best_splits = splits
                break
    return best_splits


def _sparse_attn_decode_ragged_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {main_cache.shape}"
    )
    assert main_indices.ndim == 1, (
        f"expected main_indices=[nnz], got {main_indices.shape}"
    )
    assert main_indptr.ndim == 1, f"expected main_indptr=[b+1], got {main_indptr.shape}"
    assert (
        not q.is_cpu
        and not main_cache.is_cpu
        and not main_indices.is_cpu
        and not main_indptr.is_cpu
    )

    main_indices = _as_int32_contiguous_1d(main_indices)
    main_indptr = _as_int32_contiguous_1d(main_indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert main_indptr.numel() == num_queries + 1, (
        f"expected main_indptr shape [{num_queries + 1}], got {main_indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_sparse_attn_decode_ragged_triton",
    )

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_indptr is not None
        assert extra_indices.ndim == 1, (
            f"expected extra_indices=[nnz], got {extra_indices.shape}"
        )
        assert extra_indptr.ndim == 1, (
            f"expected extra_indptr=[b+1], got {extra_indptr.shape}"
        )
        extra_indices = _as_int32_contiguous_1d(extra_indices)
        extra_indptr = _as_int32_contiguous_1d(extra_indptr)
        assert extra_indptr.numel() == num_queries + 1, (
            f"expected extra_indptr shape [{num_queries + 1}], got {extra_indptr.shape}"
        )
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    block_h = 16
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    else:
        assert out.shape == q.shape, f"expected out shape {q.shape}, got {out.shape}"
        assert out.device == q.device, (
            f"expected out on device {q.device}, got {out.device}"
        )
        assert out.dtype == torch.bfloat16, (
            f"expected out dtype {torch.bfloat16}, got {out.dtype}"
        )
    heads_blocks = triton.cdiv(num_heads, block_h)
    nope_block = triton.next_power_of_2(nope_head_dim)
    comb_dim = nope_head_dim + rope_head_dim

    block_k = 32  # KV tokens walked per split-K iteration. Tuned on gfx950.
    inv_q = 1.0 / max(1, num_queries)
    avg_main_len = main_indices.numel() * inv_q
    avg_extra_len = (extra_indices.numel() * inv_q) if has_extra else 0.0
    num_splits = _decode_num_splits(
        num_queries, heads_blocks, avg_main_len, avg_extra_len, block_k
    )

    part_m = torch.empty(
        (num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device
    )
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty(
        (num_queries, num_splits, num_heads, comb_dim),
        dtype=torch.float32,
        device=q.device,
    )

    _sparse_attn_decode_partial_kernel[(num_queries, num_splits, heads_blocks)](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        part_m,
        part_l,
        part_acc,
        q.stride(0),
        q.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        scale,
        num_heads,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=nope_block,
        ROPE_DIM=rope_head_dim,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        NUM_STAGES=1,
        num_warps=4,
    )

    _sparse_attn_decode_reduce_kernel[(num_queries, num_heads)](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        out,
        out.stride(0),
        out.stride(1),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        COMB_DIM=comb_dim,
        BLOCK_H=1,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
    return out


def _sparse_attn_decode_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    main_lengths: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    main_ragged_indices: torch.Tensor | None = None,
    main_ragged_indptr: torch.Tensor | None = None,
    extra_ragged_indices: torch.Tensor | None = None,
    extra_ragged_indptr: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if main_ragged_indices is None or main_ragged_indptr is None:
        main_ragged_indices, main_ragged_indptr = build_ragged_indices_from_dense(
            main_indices,
            main_lengths
            if main_lengths is not None
            else (main_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=main_cache.shape[0] * main_cache.shape[1],
        )

    if (
        (extra_ragged_indices is None or extra_ragged_indptr is None)
        and extra_cache is not None
        and extra_indices is not None
    ):
        extra_ragged_indices, extra_ragged_indptr = build_ragged_indices_from_dense(
            extra_indices,
            extra_lengths
            if extra_lengths is not None
            else (extra_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=extra_cache.shape[0] * extra_cache.shape[1],
        )

    return _sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_ragged_indices,
        main_indptr=main_ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_ragged_indices,
        extra_indptr=extra_ragged_indptr,
        out=out,
    )


def sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor | None,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
) -> None:
    assert kv.ndim == 3 and kv.shape[1] == 1, (
        f"SM89 Triton sparse prefill expects kv=[skv,1,d], got {kv.shape}"
    )
    _validate_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "sparse_attn_prefill",
    )

    if ragged_indices is not None and ragged_indptr is not None:
        output_chunk = _sparse_attn_prefill_ragged_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=ragged_indices,
            indptr=ragged_indptr,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
        )
    else:
        assert indices is not None
        indices_2d = indices.reshape(indices.shape[0], -1)
        output_chunk = _sparse_attn_prefill_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=indices_2d,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            topk_length=topk_length,
        )
    output.copy_(output_chunk[..., : output.shape[-1]].to(output.dtype))


def sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    assert swa_k_cache.dtype == torch.uint8, (
        "SM89 Triton sparse decode expects uint8 fp8_ds_mla SWA cache, "
        f"got {swa_k_cache.dtype}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "sparse_attn_decode",
    )

    main_indices = swa_indices.reshape(swa_indices.shape[0], -1)

    extra_cache = None
    extra_indices = None
    if not swa_only:
        assert kv_cache is not None
        assert topk_indices is not None or (
            topk_ragged_indices is not None and topk_ragged_indptr is not None
        )
        assert kv_cache.dtype == torch.uint8, (
            "SM89 Triton sparse decode expects uint8 fp8_ds_mla extra cache, "
            f"got {kv_cache.dtype}"
        )
        extra_cache = kv_cache
        if topk_indices is not None:
            extra_indices = topk_indices.reshape(topk_indices.shape[0], -1)

    direct_out = None
    attn_out = _sparse_attn_decode_triton(
        q=q,
        main_cache=swa_k_cache,
        main_indices=main_indices,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        main_lengths=swa_lens,
        extra_lengths=topk_lens,
        main_ragged_indices=swa_ragged_indices,
        main_ragged_indptr=swa_ragged_indptr,
        extra_ragged_indices=topk_ragged_indices,
        extra_ragged_indptr=topk_ragged_indptr,
        out=direct_out,
    )
    if direct_out is None:
        output.copy_(attn_out.to(output.dtype))
