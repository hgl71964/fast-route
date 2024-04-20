import sys, os

import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, sum, zeros_like


@triton.jit
def kernel(
    # Pointers to matrices
    x_ptr,
    o_ptr,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIFF: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offset = pid_m * stride_m * BLOCK_M
    k_off = tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_m +
                                 k_off[None, :])

    # shape: [BLOCK_M, 2]
    x = tl.load(x_ptrs)
    pad = tl.zeros([BLOCK_M, DIFF], dtype=tl.float32)

    tl.join
    # new = tl.cat(x, pad, False)

    o_ptrs = o_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_m +
                                 k_off[None, :])
    tl.store(o_ptrs, o)


if __name__ == '__main__':

    x = [
        [0.9, 0.5, 0.2, 0.6],
        [0.3, 0.1, 0.2, 0.2],
        [0.3, 0.9, 0.2, 0.7],
        [0.05, 0.1, 0.2, 0.002],
    ]
    b = x

    x = torch.tensor(
        x,
        dtype=torch.float16,
        device='cuda',
    )
    o = torch.empty_like(x)

    BLOCK_M = 2
    BLOCK_N = 2

    grid = (
        triton.cdiv(x.shape[0], BLOCK_M),
        triton.cdiv(x.shape[1], BLOCK_N),
    )

    DIFF = x.shape[1] - BLOCK_N

    k = kernel[grid](x, o, x.stride(0), x.stride(1), BLOCK_M, BLOCK_N, DIFF)

    # path = os.path.join(os.path.dirname(__file__), 'ttgir.mlir')
    # with open(path, 'w') as f:
    #     f.write(k.asm['ttgir'])
    #
    # path = os.path.join(os.path.dirname(__file__), 'ttir.mlir')
    # with open(path, 'w') as f:
    #     f.write(k.asm['ttir'])

    print(k.asm.keys())

    print('result: ')
    print(o)

    print('ref: ')
