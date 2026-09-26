# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Relaying V4.1 cache rows across a pipeline cut must reproduce the records
the cache writers would have stored on the receiving stage."""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA only", allow_module_level=True)

BLOCK_SIZE = 64
NUM_BLOCKS = 3
NUM_TOKENS = 40
SENTINEL = 165
# The FP4 writers convert with e2m1x2, which ptxas accepts from SM100 on.
_NEEDS_FP4 = pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="FP4 cache writers need SM100",
)


def _paged_cache(record: int, extra_page_bytes: int) -> torch.Tensor:
    page = math.ceil(BLOCK_SIZE * record / 576) * 576 + extra_page_bytes
    backing = torch.full((NUM_BLOCKS, page), SENTINEL, dtype=torch.uint8, device="cuda")
    return backing.as_strided((NUM_BLOCKS, BLOCK_SIZE, record), (page, record, 1))


def _backing(cache: torch.Tensor) -> torch.Tensor:
    return cache.as_strided((NUM_BLOCKS, cache.stride(0)), (cache.stride(0), 1))


def _kv_writer(record: int):
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        rope_quant_insert,
    )

    latent = torch.randn(NUM_TOKENS, 512, dtype=torch.bfloat16, device="cuda")
    angles = torch.randn(NUM_TOKENS, 32, device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.arange(NUM_TOKENS, dtype=torch.int64, device="cuda")

    def write(cache: torch.Tensor, slots: torch.Tensor) -> None:
        rope_quant_insert(latent, positions, cos_sin, cache, slots, 1)

    return write


def _index_k_writer(record: int):
    from vllm.models.deepseek_v41.common.ops.indexer_k_store import (
        indexer_k_norm_rope_store,
    )

    k_pre = torch.randn(NUM_TOKENS, 128, dtype=torch.bfloat16, device="cuda")
    angles = torch.randn(NUM_TOKENS, 32, device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.arange(NUM_TOKENS, dtype=torch.int64, device="cuda")
    weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")

    def write(cache: torch.Tensor, slots: torch.Tensor) -> None:
        indexer_k_norm_rope_store(
            k_pre, positions, cos_sin, weight, 1e-6, cache, slots, 1, record == 68
        )

    return write


@pytest.mark.parametrize(
    "record,make_writer",
    [
        (584, _kv_writer),
        (528, _kv_writer),
        pytest.param(288, _kv_writer, marks=_NEEDS_FP4),
        (132, _index_k_writer),
        pytest.param(68, _index_k_writer, marks=_NEEDS_FP4),
    ],
)
def test_relayed_rows_match_direct_writes(record, make_writer):
    from vllm.models.deepseek_v41.common.ops.pipeline_rows import (
        gather_paged_rows,
        scatter_paged_rows,
    )

    torch.manual_seed(0)
    write = make_writer(record)
    # Scattered over blocks and not adjacent, so a wrong value/scale split
    # would touch bytes of neighbouring records or miss the record's own.
    slots = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device="cuda")[:NUM_TOKENS]
    slots[::7] = -1

    sender = _paged_cache(record, extra_page_bytes=0)
    write(sender, slots)
    # The receiver pads its pages differently; only its records must match.
    expected = _paged_cache(record, extra_page_bytes=512)
    write(expected, slots)

    num_rows = NUM_TOKENS + 5
    rows = torch.full((num_rows, record), SENTINEL, dtype=torch.uint8, device="cuda")
    gather_paged_rows(sender, slots, rows)
    received = _paged_cache(record, extra_page_bytes=512)
    scatter_paged_rows(received, slots, rows)

    assert torch.equal(_backing(received), _backing(expected))
    # Rows without a slot, or past the slot mapping, go out as zeros.
    assert not rows[:NUM_TOKENS][slots < 0].any()
    assert not rows[NUM_TOKENS:].any()
