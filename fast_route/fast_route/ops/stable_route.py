import torch
import triton
import triton.language as tl
from triton.language.math import fdiv

from fast_route.ops.stable_argsort import argsort as stable_argsort


# yapf: disable
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128,  'BLOCK_SIZE_K': 64,}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128,  'BLOCK_SIZE_K': 32,}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64,}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32,}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 32,}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32,}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 64,}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32,}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        # triton.Config({'BLOCK_SIZE_M': 128,  'BLOCK_SIZE_K': 128,}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256,  'BLOCK_SIZE_K': 128,}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 128,}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 128,}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128,  'BLOCK_SIZE_K': 128,}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64,}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 64,}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64,}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def route_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, d_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        # Meta-parameters
        TOPK: tl.constexpr,
        DIFF: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
):
    # -----------------------------------------------------------
    pid_m = tl.program_id(axis=0)

    # ----------------------------------------------------------
    # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs,
                    mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!

    # XXX convert to fp16? may have precision issue
    # x = accumulator.to(tl.float16)
    x = accumulator

    # resolve pad (XXX: padding seems to deeply wrong here)
    # it's likely a bug in triton; individual test ok
    # for now do not have e < 16
    pad_mask = tl.arange(0, BLOCK_SIZE_N) < BLOCK_SIZE_N-DIFF
    pad_mask = tl.broadcast_to(pad_mask[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    minel = tl.broadcast_to(tl.min(x, 1)[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    x = tl.where(pad_mask, x, minel)

    # topk/sort
    ids = tl.broadcast_to(tl.arange(0, BLOCK_SIZE_N)[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    sort, sort_ids = stable_argsort(x, ids, 1, True)

    # persistent softmax
    mask = tl.arange(0, BLOCK_SIZE_N) - TOPK < 0
    mask = tl.broadcast_to(mask[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    topk_sort = tl.where(mask, sort, -float('inf'))
    x_max = tl.max(topk_sort, 1)  # [BLOCK_SIZE_M, ]
    safe_exp = tl.exp(topk_sort-x_max[:, None])
    safe_exp_sum = tl.sum(safe_exp, 1)
    c = safe_exp / safe_exp_sum[:, None]  # TODO try fast math div

    # -----------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.arange(0, BLOCK_SIZE_N)
    # offs_cn = tl.arange(0, TOPK)

    c_ptrs = c_ptr + stride_cm * offs_cm[:,
                                         None] + stride_cn * offs_cn[None, :]
    d_ptrs = d_ptr + stride_cm * offs_cm[:,
                                         None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
    tl.store(d_ptrs, sort_ids, mask=c_mask)

def fused_route(hidden_state: torch.Tensor,
                gate: torch.Tensor,
                topk: int,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                renormalize=True,
            ):
    assert renormalize, f'only support renormalize now'

    M, K = hidden_state.shape  # e.g. 512, 4096
    _, N = gate.shape         # e.g. 4096, 8
    assert N <= 128, f'{N} number of experts should be small enough to reside in shared memory'
    # assert N >= 16, f'{N} number of experts must be > 16'

    # tl.dot doesn't support tile size < 16; but gate can be padded statically
    _, E = gate.shape
    if E < 16:
        DIFF = 16 - E
        padd_gate = torch.cat([
            gate,
            torch.zeros((K, DIFF), dtype=gate.dtype, device=gate.device)
            # torch.full((K, diff), -float('inf'), dtype=gate.dtype, device=gate.device)
        ], 1)
    else:
        DIFF = 0
        padd_gate = gate

    config = {
        # 'BLOCK_SIZE_M': 16,
        # 'BLOCK_SIZE_K': 32,
        'BLOCK_SIZE_N': max(N, 16),   # entire col fit in
        'TOPK': topk,
        'DIFF': DIFF,
    }
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), )

    route_kernel[grid](
        hidden_state, padd_gate, topk_weights, topk_ids,
        M, N, K,
        hidden_state.stride(0), hidden_state.stride(1),
        gate.stride(0), gate.stride(1),
        topk_weights.stride(0), topk_weights.stride(1),
        **config,
    )

    # tl cannot directly write to ptr with incompetible shape
    topk_weights = topk_weights[:, :topk]
    topk_ids = topk_ids[:, :topk]

    # print('fused_route:')
    # print(topk_weights)
    # print(topk_ids)
    # print(topk_weights.shape, topk_ids.shape)
    # print()
    # print()
    return topk_weights, topk_ids
