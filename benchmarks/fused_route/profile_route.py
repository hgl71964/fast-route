import sys, os
import argparse

sys.path.append(os.getcwd())

import torch
import triton

from main_moe_fused import fused_moe as test_moe

from fast_route.layers.vllm_route import fused_moe as vllm_moe
from fast_route.layers.profile_vllm_route import fused_moe as profile_vllm_moe

from fast_route.layers.fused_route import fused_moe as fr_moe
from fast_route.layers.profile_fused_route import fused_moe as profile_fr_moe


def parse_args():
    parser = argparse.ArgumentParser(description="???")
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


def test_routing(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    args,
):
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    gate = torch.randn((k, e), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

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

    test_routing(args.m, args.n, args.k, args.e, args.topk, torch.float16,
                 args)
