import sys, os

import torch
import triton
import triton.language as tl
from triton.language.math import fdiv

from argsort import argsort


# yapf: disable
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
#                       num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
#                       num_warps=2),
#         # Good config for fp8 inputs.
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.jit
def renormalize_route(
        # Pointers to matrices
        a_ptr, b_ptr, topk_weight_ptr, topk_ids_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        TOPK: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
):
    pid_m = tl.program_id(axis=0)

    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)  # entire col fit in
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    accumulator = accumulator.to(tl.float16)

    # topk/sort
    ids = tl.broadcast_to(tl.arange(0, BLOCK_SIZE_N)[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    sort, sort_ids = argsort(accumulator, ids, 1, True)

    # persistent softmax
    mask = tl.arange(0, BLOCK_SIZE_N) - TOPK < 0
    mask = tl.broadcast_to(mask, (BLOCK_SIZE_M, BLOCK_SIZE_N))

    topk_sort = tl.where(mask, sort, -float('inf'))
    x_max = tl.max(topk_sort, 1)  # [BLOCK_SIZE_M, ]

    safe_exp = tl.exp(topk_sort-x_max[:, None])
    safe_exp_sum = tl.sum(safe_exp, 1)

    # ret = safe_exp / safe_exp_sum[:, None]
    ret = fdiv(safe_exp, safe_exp_sum)

    # -----------------------------------------------------------
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = tl.arange(0, TOPK)
    topk_weight_ptrs = topk_weight_ptr + (stride_cm * off_m[:, None] + stride_cn * off_n[None, :])
    topk_ids_ptrs = topk_ids_ptr + (stride_cm * off_m[:, None] + stride_cn * off_n[None, :])

    c_mask = (off_m[:, None] < M) & (off_n[None, :] < TOPK)
    tl.store(topk_weight_ptrs, ret, mask=c_mask)
    tl.store(topk_ids_ptrs, sort_ids, mask=c_mask)


def fused_route(hidden_state: torch.Tensor,
                gate: torch.Tensor,
                topk: int,
                topk_weight: torch.Tensor,
                topk_ids: torch.Tensor,
                renormalize=True,
            ):
    assert renormalize, f'only support renormalize now'

    M, K = hidden_state.shape  # e.g. 512, 4096
    KK, N = gate.shape         # e.g. 4096, 8
    assert KK == K, f'{KK}, {K}'
    assert N <= 64, f'{N} number of experts should be small enough to reside in shared memory'

    config = {
        'BLOCK_SIZE_M': 16,  # TODO autotune
        'BLOCK_SIZE_K': 32,  # TODO autotune
        'BLOCK_SIZE_N': N,   # entire col fit in
    }

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    renormalize_route[grid](
        hidden_state, gate, topk_weight, topk_ids,
        M, N, K,
        hidden_state.stride(0), hidden_state.stride(1),
        gate.stride(0), gate.stride(1),
        topk_weight.stride(0), topk_weight.stride(1),
        topk,
        **config,
    )


def f():
    # 1. mm and then each tile write only topk
    # 2. run actual torch.topk
    # 3. re-map to original index,
    pass
