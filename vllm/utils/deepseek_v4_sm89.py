# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch for the DeepSeek V4 / V4.1 Flash SM89 implementation."""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# (model_type, architecture) -> (hidden_size, num_hidden_layers) of the Flash
# checkpoints the SM89 path is sized and validated for.
_FLASH_MODELS = {
    ("deepseek_v4", "DeepseekV4ForCausalLM"): (4096, 43),
    ("deepseek_v41", "DeepseekV41ForCausalLM"): (5120, 40),
}


def is_deepseek_v4_sm89(config: "VllmConfig | None" = None) -> bool:
    if not current_platform.is_cuda() or not current_platform.is_device_capability(89):
        return False
    if config is None:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    if config is None:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if is_forward_context_available():
            context = get_forward_context()
            enabled = context.additional_kwargs.get("_deepseek_v4_sm89")
            if enabled is None:
                # Breakable graph replay invokes eager segments without model.forward.
                enabled = any(
                    getattr(layer, "_deepseek_v4_sm89", False)
                    for layer in context.no_compile_layers.values()
                )
                context.additional_kwargs["_deepseek_v4_sm89"] = enabled
            return enabled
    if config is None or config.model_config is None:
        return False
    hf_config = config.model_config.hf_config
    model_type = getattr(hf_config, "model_type", "")
    shape = (
        getattr(hf_config, "hidden_size", None),
        getattr(hf_config, "num_hidden_layers", None),
    )
    return any(
        _FLASH_MODELS.get((model_type, arch)) == shape
        for arch in getattr(hf_config, "architectures", None) or []
    )
