import torch
import triton
import triton.language as tl

import pytest

from contextlib import nullcontext

from fast_route.ops.stable_route import fused_route


def fused_moe(
    hidden_states: torch.Tensor,
    gate,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk,
    renormalize=True,
    inplace=False,
):

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

    # vanilla
    ## shape: [m, e]
    profile_ctx = nullcontext()
    ctx_pt = nullcontext()
    ctx_vllm = nullcontext()
    ctx_prep = nullcontext()
    ctx_fr = nullcontext()

    with profile_ctx as prof:

        # with record_function("torch_route"):
        with ctx_pt:
            pass

        # vllm: GEMM + fused softmax + topk
        # with record_function("vllm"):
        with ctx_vllm:
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
            # print('vllm: ', vllm_topk_weights, vllm_topk_ids)

        # tl.dot doesn't support tile size < 16; but gate can be padded statically
        with ctx_prep:
            pass

        # fused routing
        # with record_function("fused_route"):
        with ctx_fr:
            pass

            # print(id(topk_weights), id(topk_ids))
            topk_weights, topk_ids = fused_route(hidden_states, padd_gate,
                                                 topk, topk_weights, topk_ids,
                                                 renormalize)

            # print(topk_weights.shape, topk_ids.shape)
            # print(id(topk_weights), id(topk_ids))

        assert torch.allclose(ref_topk_weights,
                              topk_weights.to(ref_topk_weights.dtype),
                              atol=1e-2), (ref_topk_weights, topk_weights)

        assert torch.allclose(vllm_topk_weights, topk_weights,
                              atol=1e-2), (vllm_topk_weights, topk_weights)

        assert torch.allclose(
            vllm_topk_ids,
            ref_topk_ids,
        ), 'vllm token id mismatch'

        # NOTE: compare IDs is tricky because different tie-breaking policy
        # note that order doesn't matter, so long as expert id is included
        for m in range(M):
            for j in range(topk):
                find = False
                for k in range(topk):
                    if topk_ids[m, j] == ref_topk_ids[m, k]:
                        find = True
                        break
                if not find:
                    print(ref_topk_ids[m])
                    print(ref_topk_weights[m])
                    print()
                    print(topk_ids[m])
                    print(topk_weights[m])
                    assert False, 'mismatch token ids'


@pytest.mark.parametrize("m", [512, 1024])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("n", [8196])
@pytest.mark.parametrize("e", [8, 16])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("seed", [i for i in range(10)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("renormalize", [True])
def test_stable_route(m, k, e, n, topk, seed, renormalize, dtype):
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
    # ref_topk_weights, ref_topk_ids = torch.topk(norm, topk)

    ## CPU topk is stable?
    # ref_topk_weights, ref_topk_ids = torch.topk(norm.float().cpu(), topk)
    # ref_topk_weights, ref_topk_ids = ref_topk_weights.to('cuda'), ref_topk_ids.to('cuda')

    ## stable sort
    ref_topk_weights, ref_topk_ids = torch.sort(norm,
                                                dim=1,
                                                descending=True,
                                                stable=True)
    ref_topk_weights, ref_topk_ids = ref_topk_weights[:, :
                                                      topk], ref_topk_ids[:, :
                                                                          topk]

    ref_topk_ids = ref_topk_ids.to(torch.int32)
    if renormalize:
        ref_topk_weights = ref_topk_weights / ref_topk_weights.sum(
            dim=-1, keepdim=True)

    # -------------- fr ----------------
    topk_weights = torch.empty(
        M,
        # topk,
        E,
        dtype=dtype,
        # dtype=torch.float16,
        device=hidden_states.device)
    topk_ids = torch.empty(
        M,
        # topk,
        E,
        dtype=torch.int32,
        device=hidden_states.device)

    # invoke to auto-tune and autotune
    K, E = gate.shape
    if E < 16:
        diff = 16 - E
        padd_gate = torch.cat([
            gate,
            torch.zeros((K, diff), dtype=gate.dtype, device=gate.device)
        ], 1)
    else:
        padd_gate = gate

    fused_route(
        hidden_states,
        padd_gate,
        topk,
        topk_weights,
        topk_ids,
        renormalize,
        testing=True,
    )
    tol = {}

    # compare
    torch.testing.assert_close(ref_topk_weights, topk_weights, **tol)
    torch.testing.assert_close(ref_topk_ids, topk_ids.to(torch.int64), **tol)
