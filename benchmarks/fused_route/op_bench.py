import argparse
import torch
import triton
import time

import vllm._moe_C as moe_kernels
from fast_route.ops.stable_route import fused_route as stable_route


def parse_args():
    parser = argparse.ArgumentParser(description="???")
    parser.add_argument('-p', action='store_true', help='A boolean flag')
    parser.add_argument(
        '-m',
        type=int,
        default=512,
    )
    parser.add_argument(
        '-n',
        type=int,
        default=8192,
    )
    parser.add_argument(
        '-k',
        type=int,
        default=4096,
    )
    parser.add_argument(
        '-e',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    m = args.m
    n = args.n
    k = args.k
    e = args.e
    topk = args.topk

    torch.manual_seed(args.seed)

    def vllm_route(hidden_states, gate, topk, topk_weights, topk_ids, token_expert_indicies, renormalize):
        # vllm: GEMM + fused softmax & topk
        gating_output = hidden_states @ gate
        moe_kernels.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output.float(),  
        )
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['m'],  # Argument names to use as an x-axis for the plot
            x_vals=[2**i for i in range(0, 15)
                    ],  # Different possible values for `x_name`
            line_arg=
            'provider',  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=['fr', 'vllm'],
            # Label name for the lines
            line_names=["fast_route", "vLLM MoE layer"],

            # Line styles
            styles=[('blue', '-'), ('green', '-')],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name=
            "moe_bench",  # Name for the plot, used also as a file name for saving the plot.
            args={
                'n': n,
                'k': k,
                'e': e,
                'topk': topk,
            },
        ))
    def benchmark(m, provider, n, k, e, topk):

        m = m
        # n = 14336 // 2
        # k = 4096
        # e = 8
        # topk = 2

        # torch.cuda.manual_seed(3227)
        dtype = torch.float16

        a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
        hidden_states = a
        gate = torch.randn((k, e), device='cuda', dtype=dtype) / 10

        topk_weights = torch.empty(
            m,
            e,
            dtype=torch.float32,
            # dtype=torch.float16,
            device=hidden_states.device)
        topk_ids = torch.empty(m,
                            e,
                            dtype=torch.int32,
                            device=hidden_states.device)
        token_expert_indicies = torch.empty(m,
                                            topk,
                                            dtype=torch.int32,
                                            device=hidden_states.device)

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'fr':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: stable_route(hidden_states, gate, topk, topk_weights, topk_ids, True),
                quantiles=quantiles)
        if provider == 'vllm':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: vllm_route(hidden_states, gate, topk, topk_weights, topk_ids, token_expert_indicies, True),
                quantiles=quantiles)
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True,
                  print_data=True,
                  save_path=f'./data/op_bench_{n}_{k}_{e}_{topk}')
