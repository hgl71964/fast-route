import os
from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from torch.profiler import record_function, ProfilerActivity


class FlashAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func

        self.attn = flash_attn_func

    def forward(self, x: Tensor, profile=False) -> Tensor:
        if profile:
            ctx = torch.profiler.profile(record_shapes=True)
            ctx1 = record_function("attn_prep")
            ctx2 = record_function("attn")
            ctx3 = record_function("attn_epilogue")
        else:
            ctx = nullcontext()
            ctx1 = nullcontext()
            ctx2 = nullcontext()
            ctx3 = nullcontext()

        # TODO
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD),
                          dtype=dtype,
                          device=device,
                          requires_grad=True)
        flash_attn_func(qkv, causal=causal)

        if profile:
            save_path = os.path.join(
                './data',
                f"attn_{bsz}_{seqlen}_{k}_{self.n_head}_{self.head_dim}.json")
            prof.export_chrome_trace(save_path)

        return y


class Attention(nn.Module):

    def __init__(self, config, dtype):
        super().__init__()
        config.dim = config.k
        config.n_local_heads = config.n_head

        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head +
                          2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim,
                              total_head_dim,
                              bias=False,
                              dtype=dtype)
        self.wo = nn.Linear(config.dim, config.dim, bias=False, dtype=dtype)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self,
                x: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                input_pos: Optional[Tensor] = None,
                profile=False) -> Tensor:
        '''
        mask: [1, 1, q_seq_len, {k, v}_seq_len]
        '''
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        if profile:
            ctx = torch.profiler.profile(record_shapes=True)
            ctx1 = record_function("attn_prep")
            ctx2 = record_function("attn")
            ctx3 = record_function("attn_epilogue")
        else:
            ctx = nullcontext()
            ctx1 = nullcontext()
            ctx2 = nullcontext()
            ctx3 = nullcontext()

        with ctx as prof:
            with ctx1:
                bsz, seqlen, embed_dim = x.shape

                kv_size = self.n_local_heads * self.head_dim
                q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size],
                                             dim=-1)

                q = q.view(bsz, seqlen, self.n_head, self.head_dim)
                k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
                v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

                # RoPe preserves shape
                if freqs_cis is not None:
                    q = apply_rotary_emb(q, freqs_cis)
                    k = apply_rotary_emb(k, freqs_cis)

                q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

                if self.kv_cache is not None:
                    k, v = self.kv_cache.update(input_pos, k, v)

                k = k.repeat_interleave(self.n_head // self.n_local_heads,
                                        dim=1)
                v = v.repeat_interleave(self.n_head // self.n_local_heads,
                                        dim=1)

            with ctx2:
                # q, k, v: torch.Size([1, 32, 6, 128]), torch.Size([1, 32, 208, 128]), torch.Size([1, 32, 208, 128])
                y = F.scaled_dot_product_attention(q,
                                                   k,
                                                   v,
                                                   attn_mask=None,
                                                   is_causal=True,
                                                   dropout_p=0.0)

            with ctx3:
                y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

                y = self.wo(y)

        if profile:
            save_path = os.path.join(
                './data',
                f"attn_{bsz}_{seqlen}_{embed_dim}_{self.n_head}_{self.head_dim}.json"
            )
            prof.export_chrome_trace(save_path)

        return y


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] -
            xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] +
            xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
