# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sg_threshold = None

    def forward(self, x: Tensor, g_info: Tensor = None) -> Tensor:
        g_info_layer = g_info[0]
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, g_info[1:]


# original implementation
# class MemEffAttention(Attention):
#     def forward(self, x: Tensor, attn_bias=None) -> Tensor:
#         if not XFORMERS_AVAILABLE:
#             if attn_bias is not None:
#                 raise AssertionError("xFormers is required for using nested tensors")
#             return super().forward(x)

#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

#         q, k, v = unbind(qkv, 2)

#         x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# sg_new
# class MemEffAttention(Attention):
#     def forward(self, x: Tensor, g_info: Tensor = None, attn_bias=None) -> Tensor:
#         if not XFORMERS_AVAILABLE:
#             if attn_bias is not None:
#                 raise AssertionError("xFormers is required for using nested tensors")
#             return super().forward(x)

#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

#         q, k, v = unbind(qkv, 2)

#         if g_info is not None:
#             g_info_layer = g_info[:, 0]
#             new_g_info = g_info[:, 1:]

#             g_info_layer = g_info_layer.reshape(
#                 *g_info_layer.shape[:-1], self.num_heads, C // self.num_heads
#             )
#             q_g = g_info_layer[:, 0].unsqueeze(1)
#             k_g = g_info_layer[:, 1].unsqueeze(1)
#             v_g = g_info_layer[:, 2].unsqueeze(1)

#             num_registers = 0  # 4
#             num_reserved = 1 + num_registers

#             q_pure = q# q[:, num_reserved:]

#             q_normalized = q_pure / q_pure.norm(dim=-1, keepdim=True)
#             q_g_normalized = q_g / q_g.norm(dim=-1, keepdim=True)
#             q_sim = (q_g_normalized * q_normalized).sum(dim=-1)
#             q_sim = q_sim.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
#             q_sim = (q_sim - q_sim.min()) / (q_sim.max() - q_sim.min())
#             q_sim_pos = q_sim > 0.9  # we have ablated this from 0.6 to 0.9
#             # q_sim_pos = torch.cat(
#             #     [
#             #         q_sim_pos,
#             #         torch.zeros(size=(1, num_reserved, 1, 1), device=k.device).to(bool),
#             #     ],
#             #     dim=1,
#             # )
#             q_pos_mask = q_sim_pos.expand_as(k)
#             k_g = k_g.expand(-1, N, -1, -1)
#             v_g = v_g.expand(-1, N, -1, -1)
            
#             k[q_pos_mask] = k_g[q_pos_mask].to(k.dtype)
#             v[q_pos_mask] = v_g[q_pos_mask].to(v.dtype)

#             x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

#         else:
#             new_g_info = None
#             x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x, new_g_info


# sg_new2
# class MemEffAttention(Attention):
#     def forward(self, x: Tensor, g_info: Tensor = None, attn_bias=None) -> Tensor:
#         if not XFORMERS_AVAILABLE:
#             if attn_bias is not None:
#                 raise AssertionError("xFormers is required for using nested tensors")
#             return super().forward(x)

#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

#         q, k, v = unbind(qkv, 2)

#         if g_info is not None:
#             g_info_layer = g_info[:,0]
#             new_g_info = g_info[:,1:]

#             g_info_layer = g_info_layer.reshape(
#                 *g_info_layer.shape[:-1], self.num_heads, C // self.num_heads
#             )
#             q_g = g_info_layer[:,0].unsqueeze(1)
#             k_g = g_info_layer[:, 1].unsqueeze(1)
#             v_g = g_info_layer[:, 2].unsqueeze(1)


#             num_registers = 0  # 4
#             num_reserved = 1 + num_registers

#             q_pure = q[:, num_reserved:]
#             k_pure = k[:, num_reserved:]
#             v_pure = v[:, num_reserved:]

#             q_normalized = q_pure / q_pure.norm(dim=-1, keepdim=True)
#             q_g_normalized = q_g / q_g.norm(dim=-1, keepdim=True)
#             q_sim = (q_g_normalized * q_normalized).sum(dim=-1)
#             q_sim = q_sim.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
#             q_sim = (q_sim - q_sim.min()) / (q_sim.max() - q_sim.min())

#             q_sim_pos = q_sim > self.sg_threshold  

#             q_sim_neg = ~q_sim_pos

#             q_pos_mask = q_sim_pos.expand_as(q_pure)
#             q_neg_mask = q_sim_neg.expand_as(q_pure)

#             def get_masked_tokens(pure, token_mask, num_reserved):
#                 token_mask_flat = token_mask.permute(1, 0, 2, 3).reshape(
#                     N - num_reserved, -1
#                 )
#                 pure_flat = pure.permute(1, 0, 2, 3).reshape(N - num_reserved, -1)
#                 token = pure_flat * token_mask_flat
#                 B_to_keep = token.any(dim=1)
#                 token = token[B_to_keep]
#                 token = token.reshape(
#                     -1, B, self.num_heads, C // self.num_heads
#                 ).permute(1, 0, 2, 3)
#                 return token, torch.nonzero(B_to_keep, as_tuple=True)[0]

#             q_pos, q_pos_idxs = get_masked_tokens(q_pure, q_pos_mask, num_reserved)
#             q_neg, q_neg_idxs = get_masked_tokens(q_pure, q_neg_mask, num_reserved)
#             k_pos, _ = get_masked_tokens(k_pure, q_pos_mask, num_reserved)
#             k_neg, _ = get_masked_tokens(k_pure, q_neg_mask, num_reserved)
#             v_pos, _ = get_masked_tokens(v_pure, q_pos_mask, num_reserved)
#             v_neg, _ = get_masked_tokens(v_pure, q_neg_mask, num_reserved)
            
#             k_g = k_g.expand(-1, k_pos.shape[1], -1, -1)
#             v_g = v_g.expand(-1, v_pos.shape[1], -1, -1)
            
#             k_pos = k_g.to(k_pos.dtype)
#             v_pos = v_g.to(v_pos.dtype)

#             q_pos = torch.cat([q[:, :num_reserved], q_pos], dim=1)
#             q_neg = torch.cat([q[:, :num_reserved], q_neg], dim=1)
#             k_pos = torch.cat([k[:, :num_reserved], k_pos], dim=1)
#             k_neg = torch.cat([k[:, :num_reserved], k_neg], dim=1)
#             v_pos = torch.cat([v[:, :num_reserved], v_pos], dim=1)
#             v_neg = torch.cat([v[:, :num_reserved], v_neg], dim=1)

#             x_pos = memory_efficient_attention(q_pos, k_pos, v_pos, attn_bias=attn_bias)
#             x_neg = memory_efficient_attention(q_neg, k_neg, v_neg, attn_bias=attn_bias)
#             x = torch.zeros_like(q)

#             x[:, q_pos_idxs + num_reserved] = x_pos[:, num_reserved:]
#             x[:, q_neg_idxs + num_reserved] = x_neg[:, num_reserved:]
#             # x[:, :num_reserved] = (x_pos[:, :num_reserved] + x_neg[:, :num_reserved]) / 2
#             x[:, :num_reserved] = x_pos[:, :num_reserved]

#         else:
#             new_g_info = None
#             x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x, new_g_info














# reported in the paper
class MemEffAttention(Attention):
    def forward(self, x: Tensor, g_info: Tensor = None, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        if g_info is not None:
            g_info_layer = g_info[:,0]
            new_g_info = g_info[:,1:]

            g_info_layer = g_info_layer.reshape(
                *g_info_layer.shape[:-1], self.num_heads, C // self.num_heads
            )
            q_g = g_info_layer[:,0].unsqueeze(1)


            num_registers = 0  # 4
            num_reserved = 1 + num_registers

            q_pure = q[:, num_reserved:]
            k_pure = k[:, num_reserved:]
            v_pure = v[:, num_reserved:]

            q_normalized = q_pure / q_pure.norm(dim=-1, keepdim=True)
            q_g_normalized = q_g / q_g.norm(dim=-1, keepdim=True)
            q_sim = (q_g_normalized * q_normalized).sum(dim=-1)
            q_sim = q_sim.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
            q_sim = (q_sim - q_sim.min()) / (q_sim.max() - q_sim.min())
            q_sim_pos = q_sim > self.sg_threshold  # we have ablated this from 0.6 to 0.9
            q_sim_neg = ~q_sim_pos

            q_pos_mask = q_sim_pos.expand_as(q_pure)
            q_neg_mask = q_sim_neg.expand_as(q_pure)

            def get_masked_tokens(pure, token_mask, num_reserved):
                token_mask_flat = token_mask.permute(1, 0, 2, 3).reshape(
                    N - num_reserved, -1
                )
                pure_flat = pure.permute(1, 0, 2, 3).reshape(N - num_reserved, -1)
                token = pure_flat * token_mask_flat
                B_to_keep = token.any(dim=1)
                token = token[B_to_keep]
                token = token.reshape(
                    -1, B, self.num_heads, C // self.num_heads
                ).permute(1, 0, 2, 3)
                return token, torch.nonzero(B_to_keep, as_tuple=True)[0]

            q_pos, q_pos_idxs = get_masked_tokens(q_pure, q_pos_mask, num_reserved)
            q_neg, q_neg_idxs = get_masked_tokens(q_pure, q_neg_mask, num_reserved)
            k_pos, _ = get_masked_tokens(k_pure, q_pos_mask, num_reserved)
            k_neg, _ = get_masked_tokens(k_pure, q_neg_mask, num_reserved)
            v_pos, _ = get_masked_tokens(v_pure, q_pos_mask, num_reserved)
            v_neg, _ = get_masked_tokens(v_pure, q_neg_mask, num_reserved)

            q_pos = torch.cat([q[:, :num_reserved], q_pos], dim=1)
            q_neg = torch.cat([q[:, :num_reserved], q_neg], dim=1)
            k_pos = torch.cat([k[:, :num_reserved], k_pos], dim=1)
            k_neg = torch.cat([k[:, :num_reserved], k_neg], dim=1)
            v_pos = torch.cat([v[:, :num_reserved], v_pos], dim=1)
            v_neg = torch.cat([v[:, :num_reserved], v_neg], dim=1)

            x_pos = memory_efficient_attention(q_pos, k_pos, v_pos, attn_bias=attn_bias)
            x_neg = memory_efficient_attention(q_neg, k_neg, v_neg, attn_bias=attn_bias)
            x = torch.zeros_like(q)

            x[:, q_pos_idxs + num_reserved] = x_pos[:, num_reserved:]
            x[:, q_neg_idxs + num_reserved] = x_neg[:, num_reserved:]
            # x[:, :num_reserved] = (x_pos[:, :num_reserved] + x_neg[:, :num_reserved]) / 2
            x[:, :num_reserved] = x_pos[:, :num_reserved]

        else:
            new_g_info = None
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, new_g_info
