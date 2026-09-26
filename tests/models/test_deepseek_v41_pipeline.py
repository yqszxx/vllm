# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which DeepSeek V4.1 shared state must travel for a given pipeline cut."""

from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v41.common.pipeline import (
    get_sharing_dependencies,
    get_sharing_routes,
)


def _config():
    # DeepSeek V4.1 Flash.
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
    )


def _routes(cuts: list[int]) -> set[tuple[str, int, int]]:
    config = _config()
    bounds = [0, *cuts, config.num_hidden_layers]
    ranges = list(zip(bounds, bounds[1:]))
    routes = get_sharing_routes(config, get_sharing_dependencies(config, ranges))
    return {(r.kind, r.source_layer, r.sender) for r in routes}


@pytest.mark.parametrize("cuts", [[], [20], [8, 20], [8, 14, 20], [1, 2, 8]])
def test_cuts_at_sources_relay_nothing(cuts):
    assert _routes(cuts) == set()


def test_pp3_cut_inside_the_ratio1_group():
    # Stage 2 (27-39) reads layer 20's caches and candidates, and layer 24's
    # indices until layer 28 publishes its own.
    assert _routes([14, 27]) == {
        ("kv", 20, 1),
        ("index_k", 20, 1),
        ("index", 24, 1),
        ("candidate", 20, 1),
    }


def test_cut_right_after_the_source():
    assert _routes([21]) == {
        ("kv", 20, 0),
        ("index_k", 20, 0),
        ("index", 20, 0),
        ("candidate", 20, 0),
    }


def test_state_is_relayed_through_every_intervening_stage():
    routes = _routes([14, 20, 25, 33])
    assert {("kv", 20, 2), ("kv", 20, 3)} <= routes
    assert {("candidate", 20, 2), ("candidate", 20, 3)} <= routes
    # Stage 3 (25-32) publishes its own indices at 28 and 32.
    assert ("index", 24, 2) in routes
    assert ("index", 32, 3) in routes
    assert ("index", 24, 3) not in routes


@pytest.mark.parametrize("cuts", [[10], [14, 16], [8, 11, 27]])
def test_cuts_inside_ratio2_groups_are_rejected(cuts):
    with pytest.raises(NotImplementedError, match="compress_ratio 2"):
        _routes(cuts)
