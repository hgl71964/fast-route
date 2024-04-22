# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, os
import argparse

sys.path.append(os.getcwd())

import torch
import triton
from activation import SiluAndMul
from main_moe_fused import fused_moe
import time

from torch.profiler import profile, record_function, ProfilerActivity


def parse_args():
    parser = argparse.ArgumentParser(description="???")
    parser.add_argument('-p', action='store_true', help='A boolean flag')
    return parser.parse_args()


def run_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    args,
):
    torch.cuda.manual_seed(3227)

    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    gate = torch.randn((k, e), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    ref_out = fused_moe(a,
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

    run_moe(m, n, k, e, topk, torch.float16, parse_args())
