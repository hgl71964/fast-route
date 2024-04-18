import torch
import argparse

import triton
import triton.language as tl


@triton.jit
def onepass_softmax(
    x_ptr,
    y_ptr,
    x_stride,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
):
    pid_m = tl.program_id(0)

    m_offset = pid_m * BLOCK_M * x_stride
    k_offset = tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    y_ptrs = y_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) 
    for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
        mask = k * BLOCK_N + k_offset < x_stride
        x = tl.load(x_ptrs, mask[None, :], -float('inf'))

        # local max, sum
        m_ij = tl.maximum(tl.max(x, 1), m_i)
        x_local = tl.exp(x-m_ij[:, None])
        l_ij = tl.sum(x_local, 1)

        alpha = tl.exp(m_i - m_ij)

        # update
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # compute
        y = ...

        # store (XXX it not possible to write back only once)
        tl.store(y_ptrs, y, mask[None, :])

        x_ptrs += BLOCK_N
        y_ptrs += BLOCK_N


@triton.jit
def online_softmax(
    x_ptr,
    y_ptr,
    x_stride,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
):
    pid_m = tl.program_id(0)

    m_offset = pid_m * BLOCK_M * x_stride
    k_offset = tl.arange(0, BLOCK_N)

    # online softmax
    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) 
    for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
        mask = k * BLOCK_N + k_offset < x_stride
        x = tl.load(x_ptrs, mask[None, :], -float('inf'))

        # local max, sum
        m_ij = tl.maximum(tl.max(x, 1), m_i)
        # m_ij = tl.max(x, 1)   # NOTE <- this can work too (FLASHDECODING++, the scaling factor can be arbitrary number)
        alpha = tl.exp(m_i - m_ij)

        l_ij = tl.exp(x-m_ij[:, None])
        l_ij = tl.sum(l_ij, 1)

        # update
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        x_ptrs += BLOCK_N

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    y_ptrs = y_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
        mask = k * BLOCK_N + k_offset < x_stride
        x = tl.load(x_ptrs, mask[None, :], -float('inf'))

        numerator = tl.exp(x - m_i[:, None])
        denominator = l_i[:, None]
        y = numerator / denominator

        tl.store(y_ptrs, y, mask[None, :])

        x_ptrs += BLOCK_N
        y_ptrs += BLOCK_N


@triton.jit
def safe_softmax(
    x_ptr,
    y_ptr,
    x_stride,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
):
    pid_m = tl.program_id(0)

    m_offset = pid_m * x_stride * BLOCK_M
    k_offset = tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
        mask = k * BLOCK_N + k_offset < x_stride
        x = tl.load(x_ptrs, mask[None, :], -float('inf'))
        m_ij = tl.max(x, 1)
        m_i = tl.maximum(m_ij, m_i)
        x_ptrs += BLOCK_N

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) 
    for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
        mask = k * BLOCK_N + k_offset < x_stride
        x = tl.load(x_ptrs, mask[None, :], 0)
        l_ij = tl.exp(x - m_i[:, None])
        l_i += tl.sum(l_ij, 1)
        x_ptrs += BLOCK_N

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    y_ptrs = y_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * x_stride + k_offset[None, :])
    for k in range(0, tl.cdiv(x_stride, BLOCK_N)):
        mask = k * BLOCK_N + k_offset < x_stride
        x = tl.load(x_ptrs, mask[None, :], 0)

        numerator = tl.exp(x - m_i[:, None])
        denominator = l_i[:, None]
        y = numerator / denominator

        tl.store(y_ptrs, y, mask[None, :])

        x_ptrs += BLOCK_N
        y_ptrs += BLOCK_N


def call(x, kernel):
    BLOCK_M = 8
    BLOCK_N = 64
    grid = (triton.cdiv(x.shape[0], BLOCK_M),
            # tl.cdiv(x.shape[1], BLOCK_N),
        )
    out = torch.empty_like(x)

    kernel[grid](
        x, 
        out,

        x.stride(0),

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out


def softmax(a):
    return torch.softmax(a, dim=-1)


def test_total_time(shapes):
    device = torch.device("cuda:0")

    a_shape = shapes
    a = torch.randn(a_shape, dtype=torch.float16, device=device)
    print(f"shape: {a_shape}")
    print(f"{a.stride()=}")

    # fn = torch.compile(
    #     softmax,
    #     backend='inductor',
    # )

    torch_out = softmax(a)
    # triton_out = call(a, safe_softmax)
    triton_out = call(a, online_softmax)

    assert torch.allclose(torch_out, triton_out, atol=1e-2, rtol=0), (torch_out, triton_out)
    print('OK')




def parse_args():
    parser = argparse.ArgumentParser(description="Test the total time for tensor operations on specified shapes.")
    parser.add_argument('--shapes', nargs='+', type=int, action='store', metavar='SHAPE',
                        default=[128, 1024],
    )
    args = parser.parse_args()
    return args.shapes

if __name__ == "__main__":
    shapes = parse_args()
    test_total_time(shapes)
