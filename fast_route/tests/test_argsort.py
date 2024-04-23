import torch
import triton
import triton.language as tl

import pytest

from fast_route.ops.stable_argsort import argsort as stable_argsort
from fast_route.ops.argsort import argsort


@triton.jit
def sort_kerenl(
    # Pointers to matrices
    x_ptr,
    o_ptr,
    id_ptr,
    stride_m,
    stride_n,
    DESCEND: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STABLE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offset = pid_m * stride_m * BLOCK_M
    k_off = tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_m +
                                 k_off[None, :])

    # shape: [BLOCK_M, BLOCK_N]
    x = tl.load(x_ptrs)
    ids = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_M, BLOCK_N))

    if STABLE:
        o, ids = stable_argsort(x, ids, 1, DESCEND)
    else:
        o, ids = argsort(x, ids, 1, DESCEND)

    o_ptrs = o_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_m +
                                 k_off[None, :])
    id_ptrs = id_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_m +
                                   k_off[None, :])

    tl.store(o_ptrs, o)
    tl.store(id_ptrs, ids)


@pytest.mark.parametrize("m", [2, 8, 16, 64])
@pytest.mark.parametrize("k", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("seed", [i for i in range(10)])
@pytest.mark.parametrize("descend", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("id_dtype", [torch.int64, torch.int32])
def test_stable_argsort(m, k, seed, descend, dtype, id_dtype):
    torch.manual_seed(seed)

    x = torch.randn(
        (m, k),
        dtype=dtype,
        device='cuda',
    )
    o = torch.empty_like(x)
    ids = torch.empty(x.shape, dtype=id_dtype, device='cuda')

    BLOCK_M = 2
    BLOCK_N = k  # must fit in
    grid = (
        triton.cdiv(x.shape[0], BLOCK_M),
        triton.cdiv(x.shape[1], BLOCK_N),
    )
    STABLE = 1

    sort_kerenl[grid](x, o, ids, x.stride(0), x.stride(1), int(descend),
                      BLOCK_M, BLOCK_N, STABLE)

    # NOTE: torch.sort must set stable = True
    ref_o, ref_ids = torch.sort(x,
                                dim=1,
                                descending=bool(descend),
                                stable=True)
    tol = {}

    # compare
    torch.testing.assert_close(o, ref_o, **tol)

    ids = ids.to(id_dtype)
    ref_ids = ref_ids.to(id_dtype)  # by default, torch.int64
    torch.testing.assert_close(ids, ref_ids, **tol)


@pytest.mark.parametrize("m", [2, 8, 16, 64])
@pytest.mark.parametrize("k", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("seed", [i for i in range(10)])
@pytest.mark.parametrize("descend", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("id_dtype", [torch.int64, torch.int32])
def test_argsort(m, k, seed, descend, dtype, id_dtype):
    torch.manual_seed(seed)

    x = torch.randn(
        (m, k),
        dtype=dtype,
        device='cuda',
    )
    o = torch.empty_like(x)
    ids = torch.empty(x.shape, dtype=id_dtype, device='cuda')

    BLOCK_M = 2
    BLOCK_N = k  # must fit in
    grid = (
        triton.cdiv(x.shape[0], BLOCK_M),
        triton.cdiv(x.shape[1], BLOCK_N),
    )
    STABLE = 0

    sort_kerenl[grid](x, o, ids, x.stride(0), x.stride(1), int(descend),
                      BLOCK_M, BLOCK_N, STABLE)

    # NOTE: torch.sort must set stable = True
    ref_o, ref_ids = torch.sort(x,
                                dim=1,
                                descending=bool(descend),
                                stable=True)
    tol = {}

    # compare: all unstable argsort we only compare weights
    torch.testing.assert_close(o, ref_o, **tol)
