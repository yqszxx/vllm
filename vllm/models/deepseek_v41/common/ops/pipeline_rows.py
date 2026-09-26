# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy per-token rows of V4.1 paged caches in and out of flat buffers.

The packed caches segregate each page: all value bytes first, then all scale
bytes, ``[block_size x value_bytes][block_size x scale_bytes]`` at a page
stride of ``cache.stride(0)``. A token's record is the concatenation of its
two segments. Each side of a transfer addresses its own pages, so padding and
alignment may differ; only the record segments must agree.
"""

import torch

from vllm.triton_utils import tl, triton

# Record width -> (value bytes, scale bytes), as the cache writers store them:
# rope_quant_insert (compressed KV: V4, V4.1 MXFP8, NVFP4) and
# indexer_k_norm_rope_store (indexer K: FP8 with an fp32 scale, MXFP4).
_RECORD_SEGMENTS = {
    584: (576, 8),
    528: (512, 16),
    288: (256, 32),
    132: (128, 4),
    68: (64, 4),
}


def _segments(cache: torch.Tensor) -> tuple[int, int]:
    assert cache.dtype == torch.uint8 and cache.ndim == 3
    assert cache.stride(2) == 1 and cache.stride(1) == cache.shape[2]
    segments = _RECORD_SEGMENTS.get(cache.shape[2])
    if segments is None:
        raise NotImplementedError(
            f"No known page layout for a {cache.shape[2]}-byte cache record"
        )
    return segments


@triton.jit
def _paged_rows_kernel(
    cache,
    rows,
    slots,
    num_slots,
    PAGE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VALUE_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    VALUE_BLOCK: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    GATHER: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(slots + t, mask=t < num_slots, other=-1).to(tl.int64)
    valid = slot >= 0
    safe_slot = tl.where(valid, slot, 0)
    page = cache + (safe_slot // BLOCK_SIZE) * PAGE_STRIDE
    offset = safe_slot % BLOCK_SIZE
    row = rows + t.to(tl.int64) * (VALUE_BYTES + SCALE_BYTES)

    v = tl.arange(0, VALUE_BLOCK)
    s = tl.arange(0, SCALE_BLOCK)
    v_mask = v < VALUE_BYTES
    s_mask = s < SCALE_BYTES
    value_ptr = page + offset * VALUE_BYTES + v
    scale_ptr = page + BLOCK_SIZE * VALUE_BYTES + offset * SCALE_BYTES + s
    if GATHER:
        # Rows without a slot are sent as zeros.
        tl.store(row + v, tl.load(value_ptr, mask=v_mask & valid, other=0), v_mask)
        tl.store(
            row + VALUE_BYTES + s,
            tl.load(scale_ptr, mask=s_mask & valid, other=0),
            s_mask,
        )
    else:
        tl.store(value_ptr, tl.load(row + v, mask=v_mask), v_mask & valid)
        tl.store(scale_ptr, tl.load(row + VALUE_BYTES + s, mask=s_mask), s_mask & valid)


def _launch(
    cache: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor, gather: bool
) -> None:
    value_bytes, scale_bytes = _segments(cache)
    assert rows.dtype == torch.uint8 and rows.is_contiguous()
    assert rows.ndim == 2 and rows.shape[1] == value_bytes + scale_bytes
    assert slots.ndim == 1 and slots.is_contiguous()
    if rows.shape[0] == 0:
        return
    _paged_rows_kernel[(rows.shape[0],)](
        cache,
        rows,
        slots,
        min(slots.numel(), rows.shape[0]),
        PAGE_STRIDE=cache.stride(0),
        BLOCK_SIZE=cache.shape[1],
        VALUE_BYTES=value_bytes,
        SCALE_BYTES=scale_bytes,
        VALUE_BLOCK=triton.next_power_of_2(value_bytes),
        SCALE_BLOCK=triton.next_power_of_2(scale_bytes),
        GATHER=gather,
        num_warps=4,
    )


def gather_paged_rows(
    cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor
) -> None:
    """Copy the record at ``slots[t]`` into ``rows[t]``.

    Rows past ``slots`` or with a negative slot are zero-filled.
    """
    _launch(cache, rows, slots, gather=True)


def scatter_paged_rows(
    cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor
) -> None:
    """Write ``rows[t]`` to the record at ``slots[t]``, skipping negative slots."""
    _launch(cache, rows, slots, gather=False)
