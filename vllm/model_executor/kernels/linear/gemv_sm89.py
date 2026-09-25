# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 GEMV for DeepSeek V4 Flash on SM89 (vllm-backport)."""

import torch

from vllm.triton_utils import tl, triton

MAX_GEMV_TOKENS = 8


@triton.jit
def _bf16_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offs = tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        wv = tl.load(w_ptr + pid * stride_wn + k_offs, mask=k_mask, other=0.0).to(
            tl.float32
        )
        xv = tl.load(
            x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += xv * wv[None, :]
    out = tl.sum(acc, axis=1)
    tl.store(
        out_ptr + m_offs * stride_om + pid,
        out.to(out_ptr.dtype.element_ty),
        mask=m_mask,
    )


def should_use_triton_gemv(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :func:`bf16_gemv` applies to this call."""
    return (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.ndim == 2
        and weight.ndim == 2
        and x.shape[0] <= MAX_GEMV_TOKENS
        and x.shape[1] == weight.shape[1]
        and x.is_contiguous()
        and weight.is_contiguous()
    )


def bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``x @ weight.T`` for small M, in one launch."""
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty(m, n, dtype=out_dtype or x.dtype, device=x.device)
    _bf16_gemv_kernel[(n,)](
        x,
        weight,
        out,
        m,
        k,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_M=triton.next_power_of_2(max(m, 1)),
        BLOCK_K=2048 if m == 1 else 512,
        num_warps=8,
    )
    return out
