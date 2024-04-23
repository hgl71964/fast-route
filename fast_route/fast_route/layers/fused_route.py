import torch
import triton
import triton.language as tl
from vllm._C import ops

from fast_route.ops.stable_route import fused_route
from fast_route.ops.fused_moe import invoke_fused_moe_kernel, moe_align_block_size


def fused_moe(
    hidden_states: torch.Tensor,
    gate,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk,
    renormalize=True,
    inplace=False,
):
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
    M, _ = hidden_states.shape
    E, N, _ = w1.shape

    config = {
        'BLOCK_SIZE_M': 64,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 8
    }

    # if topk_ids.numel() <= w1.shape[0]:
    if M * topk <= w1.shape[0]:
        config = {
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 1
        }

    # NOTE: statically allocate route parameter
    topk_weights = torch.empty(
        M,
        E,
        dtype=torch.float32,
        # dtype=torch.float16,
        device=hidden_states.device)
    topk_ids = torch.empty(M,
                           E,
                           dtype=torch.int32,
                           device=hidden_states.device)

    # tl.dot doesn't support tile size < 16; but gate can be padded statically
    K, E = gate.shape
    if E < 16:
        diff = 16 - E
        padd_gate = torch.cat(
            [gate, torch.zeros((K, diff), dtype=gate.dtype)], 1)
    else:
        padd_gate = gate

    # NOTE the first time invoke will cause JIT and autotune
    topk_weights, topk_ids = fused_route(hidden_states, padd_gate, topk,
                                         topk_weights, topk_ids, renormalize)

    # topk_weights = topk_weights.to(torch.float32)
    #
    #
    # fused moe op
    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config['BLOCK_SIZE_M'], E)

    invoke_fused_moe_kernel(hidden_states, w1, intermediate_cache1,
                            topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, False,
                            topk_ids.shape[1], config)

    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3,
                            topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, True, 1,
                            config)

    if inplace:
        return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                         dim=1,
                         out=hidden_states)
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                     dim=1)
