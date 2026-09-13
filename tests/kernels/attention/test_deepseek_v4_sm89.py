# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89 Indexer logits must preserve row visibility and paged KV addressing."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.mqa_logits_sm89 import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(89),
    reason="DeepSeek V4 Flash SM89 kernels",
)


@pytest.mark.parametrize("m,n", [(1, 513), (17, 2049), (128, 8192)])
def test_prefill_logits_match_visible_fp32_scores(m, n):
    torch.manual_seed(42)
    q = torch.randn(m, 64, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(n, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(n, device="cuda") + 0.1
    weights = torch.rand(m, 64, device="cuda") / 64
    ks = torch.arange(m, device="cuda", dtype=torch.int32) % 3
    ke = torch.linspace(n // 2, n, m, device="cuda").to(torch.int32)
    # An empty row must be masked, even when its tiles take the early exit.
    if m > 1:
        ke[0] = ks[0]
    scores = torch.einsum("mhd,nd->mhn", q.float(), k.float()).relu()
    ref = (scores * weights[:, :, None]).sum(1) * scales
    visible = (torch.arange(n, device="cuda") >= ks[:, None]) & (
        torch.arange(n, device="cuda") < ke[:, None]
    )
    ref.masked_fill_(~visible, -torch.inf)
    out = fp8_mqa_logits_triton(q, (k, scales), weights, ks, ke, clean_logits=False)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
    assert torch.isneginf(out[~visible]).all()


@pytest.mark.parametrize("block_size", [64, 128, 256])
@pytest.mark.parametrize("strided", [False, True])
def test_paged_logits_follow_block_table_and_replay(block_size, strided):
    torch.manual_seed(43)
    b, h, d, nb = 2, 64, 128, 6
    storage = torch.zeros(
        nb,
        block_size * (d + 4) * (2 if strided else 1),
        device="cuda",
        dtype=torch.uint8,
    )
    flat = storage[:, : block_size * (d + 4)]
    cache = flat.view(nb, block_size, 1, d + 4)
    k = torch.randn(nb, block_size, d, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(nb, block_size, device="cuda") + 0.1
    flat[:, : block_size * d].copy_(k.view(torch.uint8).reshape(nb, -1))
    flat[:, block_size * d :].view(torch.float32).copy_(scales)
    table = torch.tensor([[3, 0, 2], [1, 4, 5]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor(
        [[block_size + 1], [3 * block_size - 1]],
        device="cuda",
        dtype=torch.int32,
    )
    q = torch.randn(b, 1, h, d, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.rand(b, h, device="cuda") / h

    def run():
        return fp8_paged_mqa_logits_triton(
            q,
            cache,
            weights,
            lengths,
            table,
            3 * block_size,
        )

    run()  # Autotune before capture.
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    lengths.zero_()  # Capture may use zero compressed context; replay can grow it.
    with torch.cuda.graph(graph):
        out = run()
    for factor, length in ((1.0, block_size + 1), (0.5, 3 * block_size - 1)):
        lengths.fill_(length)
        weights.mul_(factor)
        graph.replay()
        for i in range(b):
            keys = k[table[i].long()].reshape(-1, d).float()
            sf = scales[table[i].long()].reshape(-1)
            scores = (q[i, 0].float() @ keys.T) * sf
            ref = (scores.relu() * weights[i, :, None]).sum(0)
            ref[lengths[i, 0] :] = -torch.inf
            torch.testing.assert_close(out[i], ref, atol=2e-3, rtol=2e-3)


def test_paged_empty_context_is_masked():
    q = torch.zeros(1, 1, 64, 128, device="cuda").to(torch.float8_e4m3fn)
    cache = torch.zeros(1, 64, 1, 132, device="cuda", dtype=torch.uint8)
    weights = torch.ones(1, 64, device="cuda")
    lens = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    out = fp8_paged_mqa_logits_triton(q, cache, weights, lens, table, 64)
    assert torch.isneginf(out).all()


@pytest.fixture(autouse=True)
def sm89_model_config(default_vllm_config, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        default_vllm_config,
        "model_config",
        SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                hidden_size=4096,
                num_hidden_layers=43,
                architectures=["DeepseekV4ForCausalLM"],
            )
        ),
    )


@pytest.mark.parametrize("t", [1, 4, 128])
def test_sparse_prefill_and_decode_match_paged_reference(t):
    from tests.kernels.attention.test_rocm_triton_attn_dsv4 import (
        _pack_fp8_ds_mla_cache,
        _ragged_from_rows,
        _ref_sparse_decode_ragged,
        _ref_sparse_prefill_ragged,
    )
    from vllm.v1.attention.ops.sm89_mla_sparse import (
        _sparse_attn_decode_ragged_triton,
        _sparse_attn_prefill_ragged_triton,
    )

    torch.manual_seed(17)
    q = torch.randn(t, 8, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    rows = [torch.randperm(1024)[: 512 - i % 7].tolist() for i in range(t)]
    if t > 1:
        rows[-1] = []
    indices, indptr = _ragged_from_rows(rows, q.device)
    sink = torch.randn(8, device="cuda") * 0.1
    scale = 512**-0.5
    prefill = _sparse_attn_prefill_ragged_triton(
        q,
        kv,
        indices,
        indptr,
        scale,
        sink,
        448,
        64,
    )
    ref = _ref_sparse_prefill_ragged(q, kv, rows, scale, sink)
    torch.testing.assert_close(prefill, ref, atol=2e-3, rtol=2e-2)
    cache = _pack_fp8_ds_mla_cache(kv, 256, False)
    decode = _sparse_attn_decode_ragged_triton(
        q,
        cache,
        indices,
        indptr,
        scale,
        sink,
        448,
        64,
    )
    ref = _ref_sparse_decode_ragged(q, cache, rows, scale, sink, 256)
    torch.testing.assert_close(decode, ref, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("n", [0, 1, 17, 8191, 8192])
def test_ragged_output_buffers_preserve_lengths_and_storage(n):
    from vllm.v1.attention.ops.sm89_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    indices = torch.arange(n * 8, device="cuda", dtype=torch.int32).reshape(n, 8)
    lengths = torch.arange(n, device="cuda", dtype=torch.int32) % 9
    out = torch.full((max(n * 8, 1),), -99, device="cuda", dtype=torch.int32)
    ptr = torch.full((n + 1,), -99, device="cuda", dtype=torch.int32)
    flat, actual_ptr = build_ragged_indices_from_dense(
        indices,
        lengths,
        indices_out=out,
        indptr_out=ptr,
    )
    ref_ptr = torch.cat([lengths.new_zeros(1), lengths.cumsum(0).int()])
    torch.testing.assert_close(actual_ptr, ref_ptr)
    assert actual_ptr.data_ptr() == ptr.data_ptr()
    if n:
        assert flat.data_ptr() == out.data_ptr()
        mask = torch.arange(8, device="cuda")[None, :] < lengths[:, None]
        ref = indices[mask]
        torch.testing.assert_close(flat[: ref.numel()], ref)


@pytest.mark.parametrize("m", [1, 6, 8])
def test_small_gemv_matches_fp32_projection(m):
    from vllm.model_executor.kernels.linear.gemv_sm89 import bf16_gemv

    torch.manual_seed(31)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
    out = bf16_gemv(x, weight, torch.float32)
    torch.testing.assert_close(out, x.float() @ weight.float().T, atol=2e-3, rtol=2e-4)


def test_sm89_marlin_preserves_block_fp8_for_grouped_output_projection():
    """Loading wo_a must leave weights readable by inverse-RoPE + BF16 BMM."""
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
        MarlinFP8ScaledMMLinearKernel,
    )
    from vllm.v1.attention.ops.sm89_mla_sparse import _get_cached_wo_a_bf16

    torch.manual_seed(44)
    layer = torch.nn.Module()
    weight = torch.randn(1024, 4096, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand(8, 32, device="cuda") + 0.1
    layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scale, requires_grad=False)
    )
    layer.is_bmm = True
    kernel = MarlinFP8ScaledMMLinearKernel.__new__(MarlinFP8ScaledMMLinearKernel)
    kernel.block_quant = True
    kernel.process_weights_after_loading(layer)
    actual = _get_cached_wo_a_bf16(layer, 1, 1024, 4096)
    ref = (
        weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual[0], ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 128])
def test_sm89_broadcast_prenorm_uses_unexpanded_input(num_tokens):
    """The first layer's broadcast GEMM has hidden_size columns, not 4x."""
    from vllm.model_executor.kernels.mhc.tilelang import _hc_prenorm_gemm_outputs

    torch.manual_seed(45)
    x = torch.randn(num_tokens, 4096, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, 4096, device="cuda") / 64
    out, sqrsum = _hc_prenorm_gemm_outputs(
        x, fn, hidden_size=4096, hc_mult=4, use_tilelang_fallback=False
    )
    torch.testing.assert_close(out[0], x.float() @ fn.T, atol=0.02, rtol=0.02)
    torch.testing.assert_close(sqrsum[0], x.float().square().sum(-1))


def test_sm89_dspark_window_and_graph_replay(default_vllm_config, monkeypatch):
    """Draft blocks retain future tokens beyond the causal window after replay."""
    from types import SimpleNamespace

    from tests.kernels.attention.test_rocm_triton_attn_dsv4 import (
        _pack_fp8_ds_mla_cache,
        _ref_sparse_decode_ragged,
    )
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.models.deepseek_v4.nvidia.sm89 import DeepseekV4SM89SWAMetadataBuilder
    from vllm.v1.attention.ops.sm89_mla_sparse import _sparse_attn_decode_ragged_triton
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    config = default_vllm_config
    config.model_config.hf_config.sliding_window = 128
    config.model_config.max_model_len = 1024
    monkeypatch.setattr(
        config, "scheduler_config", SimpleNamespace(max_num_batched_tokens=10)
    )
    monkeypatch.setattr(
        config,
        "speculative_config",
        SimpleNamespace(
            num_speculative_tokens=5, parallel_drafting=True, use_dspark=lambda: True
        ),
    )
    device = torch.device("cuda")
    spec = SlidingWindowMLASpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
    )
    builder = DeepseekV4SM89SWAMetadataBuilder(spec, [], config, device)
    common = create_common_attn_metadata(
        BatchSpec([405, 405], [5, 5]), 256, device, arange_block_indices=True
    )
    common.causal = False
    common.slot_mapping[-1] = -1  # Trailing graph padding must not gather KV.
    metadata = builder.build(0, common)
    indices_ptr = metadata.decode_swa_ragged_indices.data_ptr()
    indptr_ptr = metadata.decode_swa_ragged_indptr.data_ptr()
    torch.manual_seed(46)
    q = torch.randn(10, 8, 512, device=device, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(1024, 512, device=device, dtype=torch.bfloat16) * 0.1
    cache = _pack_fp8_ds_mla_cache(kv, 256, False)
    sink = torch.randn(8, device=device) * 0.1
    scale = 512**-0.5

    def run():
        return _sparse_attn_decode_ragged_triton(
            q,
            cache,
            metadata.decode_swa_ragged_indices,
            metadata.decode_swa_ragged_indptr,
            scale,
            sink,
            448,
            64,
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for context in (12, 300, 400):
        common.seq_lens.fill_(context + 5)
        metadata = builder.build(0, common)
        assert metadata.decode_swa_ragged_indices.data_ptr() == indices_ptr
        assert metadata.decode_swa_ragged_indptr.data_ptr() == indptr_ptr
        rows = [
            list(range(req * 512 + max(context - 128, 0), req * 512 + context + 5))
            for req in range(2)
            for _ in range(5)
        ]
        rows[-1] = []
        lengths = [len(row) for row in rows]
        ref_ptr = torch.tensor([0, *lengths], device=device).cumsum(0).int()
        torch.testing.assert_close(metadata.decode_swa_ragged_indptr, ref_ptr)
        ref_indices = torch.tensor(
            [index for row in rows for index in row], device=device, dtype=torch.int32
        )
        torch.testing.assert_close(
            metadata.decode_swa_ragged_indices[: ref_indices.numel()], ref_indices
        )
        graph.replay()
        ref = _ref_sparse_decode_ragged(q, cache, rows, scale, sink, 256)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-2)
