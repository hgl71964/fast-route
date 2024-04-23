# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, os
from contextlib import nullcontext
"""Fused MoE kernel."""
import torch
import triton
import triton.language as tl
from vllm._C import ops
import vllm._moe_C as moe_kernels

from fast_route.ops.stable_route import fused_route

from torch.profiler import record_function, ProfilerActivity


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_weight,
    stride_token_id,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can be any shape representing batches and K is the feature dimension of each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is the number of experts, K is the input feature dimension, and N is the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the total number of tokens post padding, topk is the number of times each token is repeated,
        and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens, repeated topk times and arranged by the expert index they are assigned to.
    - expert_ids: A tensor containing the indices of the expert for each block. It determines which expert matrix from B should be used for each block in A.
    This kernel performs the multiplication of a token by its corresponding expert matrix as determined by `expert_ids`. The sorting of `sorted_token_ids`
    by expert index and padding ensures divisibility by BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_weight,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int,
                         num_experts: int):
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1), ),
        dtype=torch.int32,
        device=topk_ids.device)
    expert_ids = torch.empty((topk_ids.numel() + num_experts, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)

    ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int, config: dict):

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )

    # print(f"Base {config}\n")
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        topk_weights.stride(1),
        sorted_token_ids.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16,
        **config,
    )


def fused_moe(hidden_states: torch.Tensor,
              gate,
              w1: torch.Tensor,
              w2: torch.Tensor,
              topk,
              renormalize=True,
              inplace=False,
              profile=False):
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - inplace (bool): If True, perform the operation in-place. Defaults to False.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Incompatible dimensions"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    M, K = hidden_states.shape
    E, N, _ = w1.shape

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
    if profile:
        profile_ctx = torch.profiler.profile(record_shapes=True)
        ctx_pt = record_function("torch_route")
        ctx_vllm = record_function("vllm")
        ctx_prep = record_function("prep")
        ctx_fr = record_function("fused_route")
    else:
        profile_ctx = nullcontext()
        ctx_pt = nullcontext()
        ctx_vllm = nullcontext()
        ctx_prep = nullcontext()
        ctx_fr = nullcontext()

    with profile_ctx as prof:

        # with record_function("torch_route"):
        with ctx_pt:
            score = hidden_states @ gate
            norm = torch.softmax(score, dim=-1)

            # NOTE that torch.topk's tie breaking is stochastic...
            # so we use torch.sort(stable=True) to create comparable results
            # tmp_topk_weights, tmp_topk_ids = torch.topk(norm, topk)

            # tmp_topk_weights, tmp_topk_ids = torch.sort(norm,
            #                             dim=1,
            #                             descending=True,
            #                             stable=True)
            # tmp_topk_weights, tmp_topk_ids = tmp_topk_weights[:, :topk], tmp_topk_ids[:, :topk]

            tmp_topk_weights, tmp_topk_ids = torch.topk(
                norm.float().cpu(), topk)
            tmp_topk_weights, tmp_topk_ids = tmp_topk_weights.to(
                'cuda'), tmp_topk_ids.to('cuda')

            tmp_topk_ids = tmp_topk_ids.to(torch.int32)
            if renormalize:
                tmp_topk_weights = tmp_topk_weights / tmp_topk_weights.sum(
                    dim=-1, keepdim=True)

        # print('ref: ')
        # print(tmp_topk_weights, tmp_topk_ids)
        # print(tmp_topk_weights.dtype, tmp_topk_ids.dtype, tmp_topk_weights.shape,
        #       tmp_topk_ids.shape)
        # print(score)
        # print(norm)
        # print()
        # print()

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

        with ctx_prep:
            topk_weights = torch.empty(
                M,
                # topk,
                E,
                dtype=torch.float32,
                # dtype=torch.float16,
                device=hidden_states.device)
            topk_ids = torch.empty(
                M,
                # topk,
                E,
                dtype=torch.int32,
                device=hidden_states.device)

            # invoke to auto-tune and autotune
            fused_route(hidden_states, gate, topk, topk_weights, topk_ids,
                        renormalize)

        # fused routing
        # with record_function("fused_route"):
        with ctx_fr:

            # print(id(topk_weights), id(topk_ids))
            topk_weights, topk_ids = fused_route(hidden_states, gate, topk,
                                                 topk_weights, topk_ids,
                                                 renormalize)

            # print(topk_weights.shape, topk_ids.shape)
            # print(id(topk_weights), id(topk_ids))

        assert torch.allclose(tmp_topk_weights,
                              topk_weights.to(tmp_topk_weights.dtype),
                              atol=1e-2), (tmp_topk_weights, topk_weights)

        assert torch.allclose(vllm_topk_weights, topk_weights,
                              atol=1e-2), (vllm_topk_weights, topk_weights)

    if profile:
        save_path = os.path.join('./data',
                                 f"route_perf_{M}_{K}_{E}_{N}_{topk}.json")
        prof.export_chrome_trace(save_path)
