# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, os

sys.path.append(os.getcwd())

import torch
import triton
from activation import SiluAndMul
from v0_moe_fused import fused_moe as fused_moe_grouped
from fused_route import fused_moe as fused_route_moe
import time

from torch.profiler import profile, record_function, ProfilerActivity


def run_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    torch.cuda.manual_seed(3227)

    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    gate = torch.randn((k, e), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    ref_out = fused_moe_grouped(a, gate, w1, w2, topk, True, False)
    out = fused_route_moe(a, gate, w1, w2, topk, True, False)

    assert torch.allclose(ref_out, out, atol=1e-2, rtol=0), (ref_out, out)


if __name__ == '__main__':

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['m'],  # Argument names to use as an x-axis for the plot
            x_vals=[2**i for i in range(0, 10)
                    ],  # Different possible values for `x_name`
            line_arg=
            'provider',  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=['gl'],
            # Label name for the lines
            line_names=[
                "Fused MoE GEMM Kernel - Column Major", "vLLM MoE GEMM Kernel"
            ],

            # Line styles
            styles=[('blue', '-'), ('green', '-')],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name=
            "test",  # Name for the plot, used also as a file name for saving the plot.
            args={},
        ))
    def benchmark(m, provider):

        m = m
        n = 14336 // 2
        k = 4096
        e = 8
        topk = 2

        torch.cuda.manual_seed(3227)
        dtype = torch.float16

        a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

        score = torch.randn((m, e), device='cuda', dtype=dtype)
        score = torch.softmax(score, dim=-1)
        topk_weight, topk_ids = torch.topk(score, topk)

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'gl':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_moe_grouped(a, w1, w2, topk_weight, topk_ids,
                                          False),
                quantiles=quantiles)
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run(show_plots=True, print_data=True, save_path='./')

m = 512
n = 14336 // 2
k = 4096
e = 8
topk = 2
run_moe(m, n, k, e, topk, torch.float16)
