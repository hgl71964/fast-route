import argparse
import torch
import triton
import time

from fast_route.layers.vllm_route import fused_moe as vllm_moe

from fast_route.layers.stable_route import fused_moe as fr_moe
from fast_route.layers.attention import Attention


def parse_args():
    parser = argparse.ArgumentParser(description="???")
    parser.add_argument('-p', action='store_true', help='A boolean flag')
    # moe
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

    # attn
    parser.add_argument(
        '-b',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--seq',
        type=int,
        default=128,
    )
    parser.add_argument(
        '--n_head',
        type=int,
        default=32,
    )
    parser.add_argument(
        '--head_dim',
        type=int,
        default=128,
    )
    return parser.parse_args()


def fused_route_transformer(attn, x, gate, w1, w2, topk, embed_dim):
    out = attn(x, None, None, profile=False) 
    a = out.view(-1, embed_dim)
    moe_out = fr_moe(
        a,
        gate,
        w1,
        w2,
        topk,
        renormalize=True,
        inplace=False,
    )
    return moe_out


def vllm_transformer(attn, x, gate, w1, w2, topk, embed_dim):
    out = attn(x, None, None, profile=False) 
    a = out.view(-1, embed_dim)
    moe_out = vllm_moe(
        a,
        gate,
        w1,
        w2,
        topk,
        renormalize=True,
        inplace=False,
    )
    return moe_out


if __name__ == '__main__':
    args = parse_args()
    n = args.n
    k = args.k
    e = args.e
    topk = args.topk

    torch.manual_seed(args.seed)

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

        x = torch.randn((args.b, args.seq, args.k), device='cuda', dtype=dtype) / 10
        gate = torch.randn((k, e), device='cuda', dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10
        attn = Attention(args, dtype).to('cuda')
    
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'fr':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_route_transformer(attn, x, gate, w1, w2, topk, k),
                quantiles=quantiles)
        if provider == 'vllm':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: vllm_transformer(attn, x, gate, w1, w2, topk, k),
                quantiles=quantiles)
        perf = lambda ms: (2 * m * n * k + 2*args.b*args.seq*args.seq*args.n_head*args.head_dim) * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True,
                  print_data=True,
                  save_path=f'./data/moe_bench_{n}_{k}_{e}_{topk}')
