import sys, os
import argparse

sys.path.append(os.getcwd())

import torch
import triton

from fast_route.layers.vllm_route import fused_moe as vllm_moe
from fast_route.layers.profile_vllm_route import fused_moe as profile_vllm_moe

from fast_route.layers.fused_route import fused_moe as fr_moe
from fast_route.layers.profile_fused_route import fused_moe as profile_fr_moe

from fast_route.layers.attention import Attention


def parse_args():
    parser = argparse.ArgumentParser(description="???")

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


def test_routing(
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    args,
):
    # a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    x = torch.randn((args.b, args.seq, args.k), device='cuda', dtype=dtype) / 10
    a = x.view(-1, args.k)
    gate = torch.randn((k, e), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    attn = Attention(args, dtype).to('cuda')
    attn(x, None, None, profile=False)  # warmup
    attn(x, None, None, profile=True)  

    # trigger JIT and autotune
    fr_moe(
        a,
        gate,
        w1,
        w2,
        topk,
        renormalize=True,
        inplace=False,
    )
    profile_fr_moe(
        a,
        gate,
        w1,
        w2,
        topk,
        renormalize=True,
        inplace=False,
    )

    # trigger JIT and autotune
    vllm_moe(
        a,
        gate,
        w1,
        w2,
        topk,
        renormalize=True,
        inplace=False,
    )
    profile_vllm_moe(
        a,
        gate,
        w1,
        w2,
        topk,
        renormalize=True,
        inplace=False,
    )


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)

    test_routing(args.n, args.k, args.e, args.topk, torch.float16,
                 args)
