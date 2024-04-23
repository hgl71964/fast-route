import sys, os
import argparse

sys.path.append(os.getcwd())

import torch
import triton

from main_moe_fused import fused_moe as test_moe

from fast_route.layers.vllm_route import fused_moe as vllm_moe
from fast_route.layers.fused_route import fused_moe as fr_moe


def parse_args():
    parser = argparse.ArgumentParser(description="???")
    parser.add_argument('-p', action='store_true', help='A boolean flag')
    parser.add_argument('--seed',
                        type=int,
                        default=1337,
                        help='A boolean flag')
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

    _ = test_moe(a,
                 gate,
                 w1,
                 w2,
                 topk,
                 renormalize=True,
                 inplace=False,
                 profile=args.p)


if __name__ == '__main__':
    m = 512
    n = 14336 // 2
    k = 4096
    e = 16
    # e = 8
    topk = 2

    args = parse_args()

    torch.manual_seed(args.seed)

    test_routing(m, n, k, e, topk, torch.float16, args)
