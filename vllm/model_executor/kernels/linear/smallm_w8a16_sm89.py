# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-scaled FP8 GEMM shaped for the batch sizes decoding actually sees.

Marlin is the only block-scaled FP8 kernel that runs on Ada, and it is built
for compute-bound GEMMs: it launches exactly one block per SM and hides latency
with a deep ``cp.async`` pipeline. A decode step has a few dozen rows, so the
same GEMM is pure weight streaming, and one block per SM leaves a third of the
memory system idle -- a bare read of the same tensor needs 512 blocks to reach
its peak. This kernel keeps Marlin's numerics (FP8 weights widened to BF16
against a BF16 activation, FP32 accumulate) and only changes the
decomposition: narrow N tiles plus split-K, sized to fill the machine. Weights
carry either one scale per 128x128 block or, as MXFP8 does, one per row and 32
columns.

Like ``gemv_sm89``, this is not registered as a linear kernel: it is called
directly from the SM89 DeepSeek V4 Flash path.
"""

import torch

from vllm.triton_utils import tl, triton

# One launch is one M tile, so BLOCK_M rounds up to a power of two and its
# shared-memory buffers cap it here. Taller GEMMs are chunked into this many
# rows at a time.
MAX_M = 128
# Chunking is linear in M -- the chunks do not share the weight read -- so past
# a few of them dequantizing the weight once and calling a dense GEMM is
# cheaper, and a GEMM that tall has the work to absorb it.
MAX_CHUNKS = 4
# Splitting K buys blocks but costs an atomic reduction into a zeroed FP32
# buffer, which is only worth it when the N tiles alone cannot fill the SMs.
MIN_TILES_WITHOUT_SPLIT = 128
TARGET_BLOCKS = 256


@triton.jit
def _w8a16_smallm_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_sn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ATOMIC: tl.constexpr,
    ROW_SCALE_K: tl.constexpr,
    ROW_SCALES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    k_per_split = tl.cdiv(K, SPLIT_K * BLOCK_K) * BLOCK_K
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Block scales: BLOCK_K is the scale group and BLOCK_N divides it, so a
    # tile never straddles a scale boundary and the scale stays scalar.
    s_row = (pid_n * BLOCK_N) // BLOCK_K
    for k0 in tl.range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        if ROW_SCALE_K:
            # Row scales vary inside the tile, so widen the weight with them;
            # E8M0 scales keep the widened FP8 values exact in BF16.
            offs_s = k0 // ROW_SCALE_K + tl.arange(0, ROW_SCALES)
            s = tl.load(
                s_ptr + offs_n[:, None] * stride_sn + offs_s[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            w = tl.reshape(
                tl.reshape(w.to(tl.float32), (BLOCK_N, ROW_SCALES, ROW_SCALE_K))
                * s[:, :, None],
                (BLOCK_N, BLOCK_K),
            )
            acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
        else:
            s = tl.load(s_ptr + s_row * stride_sn + k0 // BLOCK_K)
            acc += tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s

    out_mask = mask_m[:, None] & mask_n[None, :]
    offs_out = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if ATOMIC:
        tl.atomic_add(offs_out, acc, mask=out_mask)
    else:
        tl.store(offs_out, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def choose_config(m: int, n: int, k: int, block_k: int = 128) -> tuple[int, ...]:
    """Pick the tile, K split and warp count that fill the SMs without paying
    for a reduction that is not needed.

    Measured on Ada at the shapes this path sees: once the N tiles alone reach
    roughly the SM count, splitting K is a net loss -- the atomics and the
    zeroed FP32 buffer cost more than the extra blocks are worth.
    """
    block_m = max(16, triton.next_power_of_2(m))
    # A tile this tall needs the extra warps to cover its rows, and a wider N
    # tile to amortise reloading the activation across them.
    num_warps = 8 if block_m >= 64 else 4
    tall = block_m >= 128
    block_n = 32 if tall or triton.cdiv(n, 32) >= MIN_TILES_WITHOUT_SPLIT else 16
    n_tiles = triton.cdiv(n, block_n)
    # The reduction a split writes is BLOCK_M x N wide, so a taller tile can
    # afford fewer splits before the atomics outweigh the extra blocks.
    target = TARGET_BLOCKS // 2 if tall else TARGET_BLOCKS
    split_k = 1
    if n_tiles < MIN_TILES_WITHOUT_SPLIT:
        while (
            n_tiles * split_k < target
            and split_k < 8
            and k % (split_k * 2 * block_k) == 0
        ):
            split_k *= 2
    return block_m, block_n, split_k, num_warps


def can_implement(m: int, n: int, k: int) -> bool:
    """Whether this kernel is the right way to run a GEMM of this shape."""
    return m <= MAX_M * MAX_CHUNKS and n % 128 == 0 and k % 128 == 0


def w8a16_block_scaled_smallm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``x @ weight.T`` for BF16 ``x`` [M, K] and FP8 ``weight`` [N, K] carrying
    one scale per 128x128 block or per row and 32 columns, in a dtype Triton
    can load (a ue8m0 checkpoint stores them as E8M0 bytes and must be widened
    first)."""
    m, k = x.shape
    n = weight.shape[0]
    if weight_scale.shape == (n, k // 32):
        row_scale_k = 32
    else:
        assert weight_scale.shape == (triton.cdiv(n, 128), triton.cdiv(k, 128))
        row_scale_k = 0
    if m > MAX_M:
        out = x.new_empty((m, n), dtype=out_dtype)
        for start in range(0, m, MAX_M):
            rows = slice(start, start + MAX_M)
            out[rows] = w8a16_block_scaled_smallm(
                x[rows], weight, weight_scale, out_dtype
            )
        return out
    block_m, block_n, split_k, num_warps = choose_config(m, n, k)
    atomic = split_k > 1
    # Only the atomic path needs a zeroed accumulator.
    out = (
        torch.zeros((m, n), device=x.device, dtype=torch.float32)
        if atomic
        else torch.empty((m, n), device=x.device, dtype=out_dtype)
    )
    _w8a16_smallm_kernel[(triton.cdiv(n, block_n), split_k)](
        x,
        weight,
        weight_scale,
        out,
        m,
        n,
        k,
        x.stride(0),
        weight.stride(0),
        weight_scale.stride(0),
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=128,
        SPLIT_K=split_k,
        ATOMIC=atomic,
        ROW_SCALE_K=row_scale_k,
        ROW_SCALES=128 // row_scale_k if row_scale_k else 1,
        num_warps=num_warps,
        num_stages=3,
    )
    return out.to(out_dtype) if atomic else out
