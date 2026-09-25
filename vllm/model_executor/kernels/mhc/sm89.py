# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 Flash SM89 prenorm, adapted from vllm-backport."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _row_sqrsum_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    K,
    BLOCK_K: tl.constexpr,
):
    """out[row] = sum(x[row].float() ** 2): the fp32 sqrsum the prenorm GEMM"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + row * stride_row + k0 + offs,
            mask=k0 + offs < K,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += x * x
    tl.store(out_ptr + row, tl.sum(acc))


def hc_prenorm_gemm_cublas(x, fn, out, sqrsum) -> None:
    """Cache BF16 projection weights and compute GEMM plus FP32 square sums."""
    assert out.shape[0] == sqrsum.shape[0] == 1
    fn_bf16 = getattr(fn, "_hc_prenorm_bf16", None)
    if fn_bf16 is None:
        fn_bf16 = fn.to(torch.bfloat16)
        fn._hc_prenorm_bf16 = fn_bf16
    out[0].copy_(x @ fn_bf16.t())
    _row_sqrsum_kernel[(x.shape[0],)](
        x, sqrsum[0], x.stride(0), x.shape[1], BLOCK_K=1024, num_warps=4
    )
