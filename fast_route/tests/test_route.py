import torch
import triton
import triton.language as tl

import pytest

import vllm._moe_C as moe_kernels

from fast_route.ops.stable_route import fused_route


@pytest.mark.parametrize("m", [128, 512, 1024])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("n", [8196])
@pytest.mark.parametrize("e", [16])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("seed", [i for i in range(10)])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("renormalize", [True])
def test_stable_route(m, k, e, n, topk, seed, renormalize, dtype):
    torch.manual_seed(seed)

    hidden_states = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    gate = torch.randn((k, e), device='cuda', dtype=dtype) / 10

    M, _ = hidden_states.shape
    E, N, = e, n
    # -------------- fr ----------------
    topk_weights = torch.empty(M, E, dtype=dtype, device=hidden_states.device)
    topk_ids = torch.empty(M,
                           E,
                           dtype=torch.int64,
                           device=hidden_states.device)

    topk_weights, topk_ids = fused_route(
        hidden_states,
        gate,
        topk,
        topk_weights,
        topk_ids,
        renormalize,
    )

    # -------------- pt ----------------
    score = hidden_states @ gate
    norm = torch.softmax(score, dim=-1)

    ## NOTE that torch.topk's tie breaking is stochastic...
    ref_topk_weights, ref_topk_ids = torch.topk(norm, topk)

    ## CPU topk is stable?
    # ref_topk_weights, ref_topk_ids = torch.topk(norm.float().cpu(), topk)
    # ref_topk_weights, ref_topk_ids = ref_topk_weights.to('cuda'), ref_topk_ids.to('cuda')

    ## stable sort
    # full_weights, full_ids = torch.sort(norm,
    #                                     dim=1,
    #                                     descending=True,
    #                                     stable=True)
    # ref_topk_weights = full_weights[:, :topk]
    # ref_topk_ids = full_ids[:, :topk]

    if renormalize:
        ref_topk_weights = ref_topk_weights / ref_topk_weights.sum(
            dim=-1, keepdim=True)

    tol = {
        'atol': 1e-2,
        'rtol': 0,
    }

    # compare
    # pytest.set_trace()
    torch.testing.assert_close(ref_topk_weights, topk_weights, **tol)

    # NOTE: due to tie-break policy there's mismatch, we have demo our tie-breaking policy in test_argsort.py
    # torch.testing.assert_close(ref_topk_ids, topk_ids, **tol)


@pytest.mark.parametrize("m", [128, 512, 1024])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("n", [8196])
@pytest.mark.parametrize("e", [16, 8])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("seed", [i for i in range(10)])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("renormalize", [True])
def test_vllm_route(m, k, e, n, topk, seed, renormalize, dtype):
    torch.manual_seed(seed)

    hidden_states = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    gate = torch.randn((k, e), device='cuda', dtype=dtype) / 10

    M, _ = hidden_states.shape
    E, N, = e, n

    # -------------- pt ----------------
    score = hidden_states @ gate
    norm = torch.softmax(score, dim=-1)

    ## NOTE that torch.topk's tie breaking is stochastic...
    ## so we use torch.sort(stable=True) to create comparable results
    ref_topk_weights, ref_topk_ids = torch.topk(norm, topk)
    if renormalize:
        ref_topk_weights = ref_topk_weights / ref_topk_weights.sum(
            dim=-1, keepdim=True)

    # -------------- vllm ----------------

    # NOTE: statically allocate route parameter
    vllm_topk_weights = torch.empty(M,
                                    topk,
                                    dtype=torch.float32,
                                    device=hidden_states.device)
    vllm_topk_ids = torch.empty(M,
                                topk,
                                dtype=torch.int32,
                                device=hidden_states.device)
    vllm_token_expert_indicies = torch.empty(M,
                                             topk,
                                             dtype=torch.int32,
                                             device=hidden_states.device)

    gating_output = hidden_states @ gate
    moe_kernels.topk_softmax(
        vllm_topk_weights,
        vllm_topk_ids,
        vllm_token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
    )
    del vllm_token_expert_indicies  # Not used. Will be used in the future.
    if renormalize:
        vllm_topk_weights = vllm_topk_weights / vllm_topk_weights.sum(
            dim=-1, keepdim=True)

    tol = {
        'atol': 1e-2,
        'rtol': 0,
    }

    # compare
    torch.testing.assert_close(ref_topk_weights,
                               vllm_topk_weights.to(torch.float16), **tol)
    # NOTE: due to tie-break policy there's mismatch
    # torch.testing.assert_close(ref_topk_ids, vllm_topk_ids.to(torch.int64),
    #                            **tol)
