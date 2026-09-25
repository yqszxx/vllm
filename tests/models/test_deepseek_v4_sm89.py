# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.nvidia.sm89 import DeepseekV4SM89Attention


class _WeightOnlyLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


class _Compressor(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.fused_wkv_wgate = _WeightOnlyLinear(weight)


class _IndexerWeightsProjection(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden_states.sum(dim=-1, keepdim=True), None


class _Indexer(torch.nn.Module):
    def __init__(self, compressor_weight: torch.Tensor) -> None:
        super().__init__()
        self.compressor = _Compressor(compressor_weight)
        self.weights_proj = _IndexerWeightsProjection()


def test_sm89_input_fusion_preserves_outputs_and_weight_refit(
    monkeypatch,
    default_vllm_config,
):
    """The three projections retain parameter aliases and only fuse once."""
    from types import SimpleNamespace

    import pytest

    from vllm.platforms import current_platform

    if not current_platform.is_cuda() or not current_platform.is_device_capability(89):
        pytest.skip("SM89 required")
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
    torch.manual_seed(7)
    main_weight = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
    indexer_weight = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
    attention = DeepseekV4SM89Attention.__new__(DeepseekV4SM89Attention)
    torch.nn.Module.__init__(attention)
    attention.compressor = _Compressor(main_weight)
    attention.indexer = _Indexer(indexer_weight)
    attention.indexer.weights_proj = _WeightOnlyLinear(
        torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    )
    attention.hidden_size = 4096
    attention.fused_input_weight = None
    attention.fused_input_splits = []
    attention.aux_stream_list = None
    attention.ln_events = [None] * 4
    monkeypatch.setattr(
        DeepseekV4SM89Attention,
        "_fused_wqa_wkv_gemm",
        lambda self, hidden: hidden[:, :2],
    )
    params = [
        attention.compressor.fused_wkv_wgate.weight,
        attention.indexer.compressor.fused_wkv_wgate.weight,
        attention.indexer.weights_proj.weight,
    ]
    attention.fuse_input_gemm_weights()
    ptr = attention.fused_input_weight.data_ptr()
    attention.fuse_input_gemm_weights()
    assert attention.fused_input_weight.data_ptr() == ptr
    assert all(p.untyped_storage().data_ptr() == ptr for p in params)
    with torch.no_grad():
        params[0].mul_(0.5)  # Refit through the original parameter must stay visible.
    for n in (1, 17, 128):
        x = torch.randn(n, 4096, device="cuda", dtype=torch.bfloat16)
        out = attention._run_parallel_input_projections(x)
        for actual, weight in zip(out[1:], params):
            ref = torch.mm(x, weight.T, out_dtype=torch.float32)
            torch.testing.assert_close(actual, ref, rtol=2e-4, atol=1e-3)


def test_sm89_dispatch_excludes_other_models_and_architectures(monkeypatch):
    """Shared helpers must not change V4.1, Vision, or other GPU dispatch."""
    from types import SimpleNamespace

    from vllm.platforms import current_platform
    from vllm.utils.deepseek_v4_sm89 import is_deepseek_v4_sm89

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    cases = [
        (89, "deepseek_v4", "DeepseekV4ForCausalLM", 4096, True),
        (86, "deepseek_v4", "DeepseekV4ForCausalLM", 4096, False),
        (90, "deepseek_v4", "DeepseekV4ForCausalLM", 4096, False),
        (89, "deepseek_v41", "DeepseekV41ForCausalLM", 4096, False),
        (89, "deepseek_v4", "DeepseekV4ForConditionalGeneration", 4096, False),
        (89, "deepseek_v4", "DeepseekV4ForCausalLM", 7168, False),
    ]
    for capability, model_type, architecture, hidden, expected in cases:
        monkeypatch.setattr(
            current_platform,
            "is_device_capability",
            lambda required, actual=capability: required == actual,
        )
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type=model_type,
                    architectures=[architecture],
                    hidden_size=hidden,
                    num_hidden_layers=43,
                )
            )
        )
        assert is_deepseek_v4_sm89(config) is expected


def test_sm89_forward_restores_config_context(monkeypatch, default_vllm_config):
    """Profiling and serving enter forward after the loader's context has exited."""
    from types import SimpleNamespace

    import pytest

    from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM
    from vllm.platforms import current_platform
    from vllm.utils.deepseek_v4_sm89 import is_deepseek_v4_sm89

    if not current_platform.is_cuda() or not current_platform.is_device_capability(89):
        pytest.skip("SM89 required")
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

    class Inner(torch.nn.Module):
        def forward(self, ids, positions, intermediate, embeds):
            assert is_deepseek_v4_sm89()
            return ids

    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model._sm89_config = default_vllm_config
    model.model = Inner()
    ids = torch.tensor([1])
    with set_current_vllm_config(None):
        assert get_current_vllm_config_or_none() is None
        assert model(ids, ids) is ids
        assert get_current_vllm_config_or_none() is None


def test_sm89_graph_replay_dispatch_uses_forward_context(monkeypatch):
    """An eager graph segment can run after the model config context has exited."""
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.forward_context import override_forward_context
    from vllm.platforms import current_platform
    from vllm.utils.deepseek_v4_sm89 import is_deepseek_v4_sm89

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "is_device_capability", lambda cap: cap == 89)
    with set_current_vllm_config(None):
        for enabled in (True, False):
            context = SimpleNamespace(
                additional_kwargs={},
                no_compile_layers={"attn": SimpleNamespace(_deepseek_v4_sm89=enabled)},
            )
            with override_forward_context(context):
                assert is_deepseek_v4_sm89() is enabled
                assert is_deepseek_v4_sm89() is enabled
        with override_forward_context(None):
            assert not is_deepseek_v4_sm89()
