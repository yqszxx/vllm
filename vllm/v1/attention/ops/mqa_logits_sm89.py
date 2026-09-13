# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 Flash Indexer logits, adapted from vllm-backport fdd92607a3.

Prefill uses BF16 dot products; paged decode converts FP8 through a BF16 LUT.
Only the SM89 default path is retained, without TP query sharding.
"""

import functools

import torch

from vllm.triton_utils import tl, triton

_PAGED_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=ns) for ns in (2, 4)
]
_PREFILL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=ns) for ns in (2, 4)
]
_PREFILL_WARMUP_M = 8
_PREFILL_WARMUP_N = 8192


@functools.cache
def _get_e4m3fn_bf16_lut(device: torch.device) -> torch.Tensor:
    lut = (
        torch.arange(256, dtype=torch.uint8, device=device)
        .view(torch.float8_e4m3fn)
        .to(torch.bfloat16)
    )
    lut[0x7F] = 480.0
    lut[0xFF] = -480.0
    return lut


@triton.jit
def _decode_e4m3fn_bf16_lut(u, lut_ptr):
    return tl.load(lut_ptr + u.to(tl.uint32))


@triton.autotune(
    configs=_PAGED_AUTOTUNE_CONFIGS,
    key=["num_heads", "head_dim", "block_size"],
)
@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_fp8_ptr,
    kv_scale_ptr,
    weights_ptr,
    fp8_lut_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_kvf_block,
    stride_kvf_s,
    stride_kvf_d,
    stride_kvs_block,
    stride_kvs_s,
    stride_w_t,
    stride_w_h,
    stride_bt_b,
    stride_bt_k,
    stride_l_t,
    stride_l_n,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_IS_BF16: tl.constexpr,
):
    token_id = tl.program_id(0)
    block_rk = tl.program_id(1)

    batch_id = token_id // next_n
    next_n_id = token_id % next_n

    context_len = tl.load(context_lens_ptr + batch_id)
    if block_rk * block_size >= context_len:
        return

    q_offset = context_len - next_n + next_n_id

    block_idx = tl.load(
        block_tables_ptr + batch_id * stride_bt_b + block_rk * stride_bt_k
    ).to(tl.int64)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim
    mask_n = offs_n < block_size

    q_base = q_ptr + batch_id * stride_q_b + next_n_id * stride_q_n
    q_offs = offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
    q_mask = mask_h[:, None] & mask_d[None, :]
    if Q_IS_BF16:
        q = tl.load(q_base + q_offs, mask=q_mask, other=0.0)
    else:
        q = _decode_e4m3fn_bf16_lut(
            tl.load(q_base + q_offs, mask=q_mask, other=0), fp8_lut_ptr
        )

    kvf_base = kv_fp8_ptr + block_idx * stride_kvf_block
    k_byte = tl.load(
        kvf_base + offs_n[:, None] * stride_kvf_s + offs_d[None, :] * stride_kvf_d,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0,
    )
    kvs_base = kv_scale_ptr + block_idx * stride_kvs_block
    k_scale = tl.load(
        kvs_base + offs_n * stride_kvs_s,
        mask=mask_n,
        other=0.0,
    )
    k = _decode_e4m3fn_bf16_lut(k_byte, fp8_lut_ptr)
    s = tl.dot(q, tl.trans(k)) * k_scale[None, :]

    w = tl.load(
        weights_ptr + token_id * stride_w_t + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    s = tl.where(s > 0, s, 0.0) * w[:, None]
    out = tl.sum(s, axis=0)

    k_offset = block_rk * block_size + offs_n
    out = tl.where(k_offset <= q_offset, out, float("-inf"))

    tl.store(
        logits_ptr + token_id * stride_l_t + k_offset * stride_l_n,
        out,
        mask=mask_n & (k_offset < context_len),
    )


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of DeepGEMM's fp8_paged_mqa_logits.

    Args:
        q:             [B, next_n, H, D] fp8_e4m3fn
        kv_cache:      [num_blocks, block_size, 1, D+4] uint8 (FP8 + fp32 scale)
        weights:       [B*next_n, H] float32
        context_lens:  [B] int32
        block_tables:  [B, max_blocks] int32
        max_model_len: output width. Caller passes the active batch max so
            the logits buffer and grid stay tight.
        clean_logits: when False, skip the -inf pre-fill of the output
            (indexer top-k reads only `[:context_len]` per row).
    Returns:
        logits:        [B*next_n, max_model_len] float32
    """
    B, next_n, num_heads, head_dim = q.shape
    assert next_n == 1, "SM89 Indexer currently supports non-speculative decoding"
    assert context_lens.numel() == B
    context_lens = context_lens.reshape(-1)
    _, block_size, one, d_plus_4 = kv_cache.shape
    assert one == 1
    assert d_plus_4 == head_dim + 4

    num_blocks = kv_cache.shape[0]
    kv_flat = kv_cache.view(num_blocks, -1)
    k_end = block_size * head_dim
    kv_byte = kv_flat[:, :k_end].as_strided(
        (num_blocks, block_size, head_dim),
        (kv_flat.stride(0), head_dim, 1),
    )
    kv_scale = kv_flat[:, k_end:].view(torch.float32)
    q_byte = q.view(torch.uint8)
    q_is_bf16 = False

    if clean_logits:
        logits = torch.full(
            (B * next_n, max_model_len),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        logits = torch.empty(
            (B * next_n, max_model_len), dtype=torch.float32, device=q.device
        )

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_N = triton.next_power_of_2(block_size)

    fp8_lut = _get_e4m3fn_bf16_lut(q.device)
    if q_is_bf16:
        q_in = fp8_lut.index_select(0, q_byte.reshape(-1).to(torch.int32)).view(
            q_byte.shape
        )
    else:
        q_in = q_byte
    num_block_cols = min(block_tables.shape[1], triton.cdiv(max_model_len, block_size))
    grid = (B * next_n, num_block_cols)
    _fp8_paged_mqa_logits_kernel[grid](
        q_in,
        kv_byte,
        kv_scale,
        weights,
        fp8_lut,
        context_lens,
        block_tables,
        logits,
        q_in.stride(0),
        q_in.stride(1),
        q_in.stride(2),
        q_in.stride(3),
        kv_byte.stride(0),
        kv_byte.stride(1),
        kv_byte.stride(2),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        Q_IS_BF16=q_is_bf16,
    )
    return logits


@triton.autotune(
    configs=_PREFILL_AUTOTUNE_CONFIGS,
    key=["num_heads", "head_dim"],
)
@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    ks_ptr,
    ke_ptr,
    logits_ptr,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_w_m,
    stride_w_h,
    stride_l_m,
    stride_l_n,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    N,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_GROUP: tl.constexpr,
    FACTOR_K_SCALE: tl.constexpr,
):
    m = tl.program_id(0)
    group = tl.program_id(1)

    group_start = group * (KV_GROUP * BLOCK_N)
    ks = tl.load(ks_ptr + m)
    ke = tl.load(ke_ptr + m)

    if (group_start >= ke) | (group_start + KV_GROUP * BLOCK_N <= ks):
        offs_span = group_start + tl.arange(0, BLOCK_N)
        for _ in tl.static_range(KV_GROUP):
            tl.store(
                logits_ptr + m * stride_l_m + offs_span * stride_l_n,
                tl.full([BLOCK_N], float("-inf"), dtype=tl.float32),
                mask=offs_span < N,
            )
            offs_span += BLOCK_N
        return

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim

    q = tl.load(
        q_ptr
        + m * stride_q_m
        + offs_h[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0.0,
    )
    w = tl.load(
        weights_ptr + m * stride_w_m + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )

    for g in tl.static_range(KV_GROUP):
        n_start = group_start + g * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        if (n_start >= ke) | (n_start + BLOCK_N <= ks):
            tl.store(
                logits_ptr + m * stride_l_m + offs_n * stride_l_n,
                tl.full([BLOCK_N], float("-inf"), dtype=tl.float32),
                mask=mask_n,
            )
        else:
            k = tl.load(
                k_ptr + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )
            k_scale = tl.load(k_scale_ptr + offs_n, mask=mask_n, other=0.0)
            s = tl.dot(q, tl.trans(k))

            if FACTOR_K_SCALE:
                s = tl.where(s > 0, s, 0.0) * w[:, None]
                out = tl.sum(s, axis=0) * k_scale
            else:
                s = s * k_scale[None, :]
                s = tl.where(s > 0, s, 0.0) * w[:, None]
                out = tl.sum(s, axis=0)

            out = tl.where((offs_n >= ks) & (offs_n < ke), out, float("-inf"))

            tl.store(
                logits_ptr + m * stride_l_m + offs_n * stride_l_n,
                out,
                mask=mask_n,
            )


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of DeepGEMM's fp8_mqa_logits.

    Args:
        q:            [M, H, D] fp8_e4m3fn
        kv:           (k_fp8 [N, D], k_scales [N]) — fp8_e4m3fn, float32
        weights:      [M, H] float32
        cu_seqlen_ks: [M] int32
        cu_seqlen_ke: [M] int32
        clean_logits: when False, skip the -inf pre-fill of the output
            (indexer top-k reads only `[ks, ke)` per row). Matches DeepGEMM.
    Returns:
        logits:       [M, N] float32
    """
    return _fp8_mqa_logits_triton_impl(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        1,
    )


def _fp8_mqa_logits_triton_impl(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    kv_group: int,
) -> torch.Tensor:
    k_fp8, k_scales = kv
    k_scales = k_scales.reshape(-1)

    M, num_heads, head_dim = q.shape
    N = k_fp8.shape[0]

    logits = torch.empty((M, N), dtype=torch.float32, device=q.device)

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)

    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k_fp8.to(torch.bfloat16)

    grid = lambda meta: (  # noqa: E731
        M,
        triton.cdiv(N, meta["BLOCK_N"] * meta["KV_GROUP"]),
    )
    _fp8_mqa_logits_kernel[grid](
        q_bf16,
        k_bf16,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q_bf16.stride(0),
        q_bf16.stride(1),
        q_bf16.stride(2),
        k_bf16.stride(0),
        k_bf16.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        N=N,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        KV_GROUP=kv_group,
        FACTOR_K_SCALE=True,
    )
    return logits


@functools.cache
def warmup_mqa_logits(num_heads: int, head_dim: int, device: torch.device) -> None:
    """Prime autotuning before capture, once per model shape and device."""
    m, n = _PREFILL_WARMUP_M, _PREFILL_WARMUP_N
    q = torch.zeros(m, num_heads, head_dim, device=device).to(torch.float8_e4m3fn)
    k = torch.zeros(n, head_dim, device=device).to(torch.float8_e4m3fn)
    scales = torch.ones(n, dtype=torch.float32, device=device)
    weights = torch.ones(m, num_heads, dtype=torch.float32, device=device)
    ks = torch.zeros(m, dtype=torch.int32, device=device)
    ke = torch.full((m,), n, dtype=torch.int32, device=device)
    fp8_mqa_logits_triton(q, (k, scales), weights, ks, ke)
    for block_size in (64, 128, 256):
        cache = torch.zeros(
            2, block_size, 1, head_dim + 4, dtype=torch.uint8, device=device
        )
        lens = torch.full((1,), block_size, dtype=torch.int32, device=device)
        table = torch.zeros(1, 1, dtype=torch.int32, device=device)
        fp8_paged_mqa_logits_triton(
            q[:1, None], cache, weights[:1], lens, table, block_size
        )
