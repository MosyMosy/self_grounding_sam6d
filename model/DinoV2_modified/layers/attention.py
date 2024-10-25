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
            g_info_layer = g_info[0]
            new_g_info = g_info[1:]

            g_info_layer = g_info_layer.reshape(
                g_info_layer.shape[0], self.num_heads, C // self.num_heads
            )
            q_g = g_info_layer[0].unsqueeze(0).unsqueeze(0)

            q_pure = q[:, 5:]
            k_pure = k[:, 5:]
            v_pure = v[:, 5:]

            q_normalized = q_pure / q_pure.norm(dim=-1, keepdim=True)
            q_g_normalized = q_g / q_g.norm(dim=-1, keepdim=True)
            q_sim = (q_g_normalized * q_normalized).sum(dim=-1)
            q_sim = q_sim.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
            q_sim = (q_sim - q_sim.min()) / (q_sim.max() - q_sim.min())
            q_sim_pos = q_sim > 0.9
            q_sim_neg = ~q_sim_pos

            q_pos = q_pure[q_sim_pos.expand_as(q_pure)].reshape(
                B, -1, self.num_heads, C // self.num_heads
            )
            q_neg = q_pure[q_sim_neg.expand_as(q_pure)].reshape(
                B, -1, self.num_heads, C // self.num_heads
            )
            k_pos = k_pure[q_sim_pos.expand_as(k_pure)].reshape(
                B, -1, self.num_heads, C // self.num_heads
            )
            k_neg = k_pure[q_sim_neg.expand_as(k_pure)].reshape(
                B, -1, self.num_heads, C // self.num_heads
            )
            v_pos = v_pure[q_sim_pos.expand_as(v_pure)].reshape(
                B, -1, self.num_heads, C // self.num_heads
            )
            v_neg = v_pure[q_sim_neg.expand_as(v_pure)].reshape(
                B, -1, self.num_heads, C // self.num_heads
            )

            q_pos = torch.cat([q[:, :5], q_pos], dim=1)
            q_neg = torch.cat([q[:, :5], q_neg], dim=1)
            k_pos = torch.cat([k[:, :5], k_pos], dim=1)
            k_neg = torch.cat([k[:, :5], k_neg], dim=1)
            v_pos = torch.cat([v[:, :5], v_pos], dim=1)
            v_neg = torch.cat([v[:, :5], v_neg], dim=1)

            x_pos = memory_efficient_attention(q_pos, k_pos, v_pos, attn_bias=attn_bias)
            x_neg = memory_efficient_attention(q_neg, k_neg, v_neg, attn_bias=attn_bias)
            x = torch.zeros_like(q)
            x[:, 5:][q_sim_pos.expand_as(q_pure)] = x_pos[:, 5:].flatten()
            x[:, 5:][q_sim_neg.expand_as(q_pure)] = x_neg[:, 5:].flatten()
            x[:, :5] = (x_pos[:, :5] + x_neg[:, :5]) / 2

        else:
            new_g_info = None
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, new_g_info


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
#             g_info_layer = g_info[0]
#             new_g_info = g_info[1:]

#             g_info_layer = g_info_layer.reshape(
#                 g_info_layer.shape[0], self.num_heads, C // self.num_heads
#             )
#             q_g = g_info_layer[0].unsqueeze(0).unsqueeze(0)
#             k_g = g_info_layer[1].unsqueeze(0).unsqueeze(0)
#             v_g = g_info_layer[2].unsqueeze(0).unsqueeze(0)

#             q_normalized = q / q.norm(dim=-1, keepdim=True)
#             q_g_normalized = q_g / q_g.norm(dim=-1, keepdim=True)
#             k_normalized = k / k.norm(dim=-1, keepdim=True)
#             k_g_normalized = k_g / k_g.norm(dim=-1, keepdim=True)
#             v_normalized = v / v.norm(dim=-1, keepdim=True)
#             v_g_normalized = v_g / v_g.norm(dim=-1, keepdim=True)

#             q_sim = (q_g_normalized * q_normalized).sum(dim=-1)
#             q_sim = q_sim.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
#             k_sim = (k_g_normalized * k_normalized).sum(dim=-1)
#             k_sim = k_sim.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
#             v_sim = (v_g_normalized * v_normalized).sum(dim=-1)
#             v_sim = v_sim.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)

#             q_sim = (q_sim - q_sim.min()) / (q_sim.max() - q_sim.min())
#             k_sim = (k_sim - k_sim.min()) / (k_sim.max() - k_sim.min())
#             v_sim = (v_sim - v_sim.min()) / (v_sim.max() - v_sim.min())

#             sim = (q_sim + k_sim + v_sim) / 3
#             sim_pos = sim > 0.65
#             sim_neg = ~sim_pos

#             q_pos = q[sim_pos.expand_as(q)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )
#             q_neg = q[sim_neg.expand_as(q)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )
#             k_pos = k[sim_pos.expand_as(k)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )
#             k_neg = k[sim_neg.expand_as(k)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )
#             v_pos = v[sim_pos.expand_as(v)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )
#             v_neg = v[sim_neg.expand_as(v)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )

#             x_pos = memory_efficient_attention(q_pos, k_pos, v_pos, attn_bias=attn_bias)
#             x_neg = memory_efficient_attention(q_neg, k_neg, v_neg, attn_bias=attn_bias)
#             x = torch.zeros_like(q)
#             x[sim_pos.expand_as(x)] = x_pos.flatten()
#             x[sim_neg.expand_as(x)] = x_neg.flatten()

#         else:
#             new_g_info = None
#             x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x, new_g_info


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
#             g_info_layer = g_info[0]
#             new_g_info = g_info[1:]

#             g_info_layer = g_info_layer.reshape(
#                 g_info_layer.shape[0], self.num_heads, C // self.num_heads
#             )
#             q_g = g_info_layer[0].unsqueeze(0).unsqueeze(0)
#             q_normalized = q / q.norm(dim=-1, keepdim=True)
#             q_g_normalized = q_g / q_g.norm(dim=-1, keepdim=True)
#             q_sim = (q_g_normalized * q_normalized).sum(dim=-1)
#             q_sim = q_sim.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
#             q_sim = (q_sim - q_sim.min()) / (q_sim.max() - q_sim.min())
#             q_sim_pos = q_sim > 0.9
#             k = k * q_sim_pos

#             x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

#         else:
#             new_g_info = None
#             x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x, new_g_info


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
#             g_info_layer = g_info[0]
#             new_g_info = g_info[1:]

#             g_info_layer = g_info_layer.reshape(
#                 g_info_layer.shape[0], self.num_heads, C // self.num_heads
#             )
#             q_g = g_info_layer[0].unsqueeze(0).unsqueeze(0)
#             q_normalized = q / q.norm(dim=-1, keepdim=True)
#             q_g_normalized = q_g / q_g.norm(dim=-1, keepdim=True)
#             q_sim = (q_g_normalized * q_normalized).sum(dim=-1)
#             q_sim = q_sim.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
#             q_sim = (q_sim - q_sim.min()) / (q_sim.max() - q_sim.min())
#             # q_sim = (q_sim - q_sim.mean(1).unsqueeze(1)) / (q_sim.std(1).unsqueeze(1) + 1e-6)
#             q_sim_pos = q_sim > 0.9
#             q_sim_neg = ~q_sim_pos


#             q_pos = q[q_sim_pos.expand_as(q)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )
#             q_neg = q[q_sim_neg.expand_as(q)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )
#             k_pos = k[q_sim_pos.expand_as(k)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )
#             k_neg = k[q_sim_neg.expand_as(k)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )
#             v_pos = v[q_sim_pos.expand_as(v)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )
#             v_neg = v[q_sim_neg.expand_as(v)].reshape(
#                 B, -1, self.num_heads, C // self.num_heads
#             )

#             x_pos = memory_efficient_attention(q_pos, k_pos, v_pos, attn_bias=attn_bias)
#             x_neg = memory_efficient_attention(q_neg, k_neg, v_neg, attn_bias=attn_bias)
#             x = torch.zeros_like(q)
#             x[q_sim_pos.expand_as(x)] = x_pos.flatten()
#             x[q_sim_neg.expand_as(x)] = x_neg.flatten()

#         else:
#             new_g_info = None
#             x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x, new_g_info


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
#             g_info_layer = g_info[0]
#             new_g_info = g_info[1:]

#             g_info_layer = g_info_layer.reshape(
#                 g_info_layer.shape[0], self.num_heads, C // self.num_heads
#             )
#             q_g = g_info_layer[0].unsqueeze(0).unsqueeze(0)
#             k_g = g_info_layer[1].unsqueeze(0).unsqueeze(0)
#             v_g = g_info_layer[2].unsqueeze(0).unsqueeze(0)
#             t_g = g_info_layer[3].unsqueeze(0).unsqueeze(0)

#             q_normalized = q / q.norm(dim=-1, keepdim=True)
#             q_g_normalized = q_g / q_g.norm(dim=-1, keepdim=True)
#             q_sim = (q_g_normalized * q_normalized).sum(dim=-1).unsqueeze(-1)

#             k_normalized = k / k.norm(dim=-1, keepdim=True)
#             k_g_normalized = k_g / k_g.norm(dim=-1, keepdim=True)
#             k_sim = (k_g_normalized * k_normalized).sum(dim=-1).unsqueeze(-1)

#             v_normalized = v / v.norm(dim=-1, keepdim=True)
#             v_g_normalized = v_g / v_g.norm(dim=-1, keepdim=True)
#             v_sim = (v_g_normalized * v_normalized).sum(dim=-1).unsqueeze(-1)

#             q_sim_scale = (q_sim - q_sim.min()) / (q_sim.max() - q_sim.min())
#             k_sim_scale = (k_sim - k_sim.min()) / (k_sim.max() - k_sim.min())
#             v_sim_scale = (v_sim - v_sim.min()) / (v_sim.max() - v_sim.min())

#             # v = v * (q_sim_scale + k_sim_scale)

#             x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

#         else:
#             new_g_info = None
#             x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x, new_g_info


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
#             g_info_layer = g_info[0]
#             new_g_info = g_info[1:]

#             g_info_layer = g_info_layer.reshape(
#                 g_info_layer.shape[0], self.num_heads, C // self.num_heads
#             )
#             q_g = g_info_layer[0].unsqueeze(0).unsqueeze(0)
#             k_g = g_info_layer[1].unsqueeze(0).unsqueeze(0)
#             v_g = g_info_layer[2].unsqueeze(0).unsqueeze(0)
#             t_g = g_info_layer[3].unsqueeze(0).unsqueeze(0)

#             q_normalized = q / q.norm(dim=-1, keepdim=True)
#             q_g_normalized = q_g / q_g.norm(dim=-1, keepdim=True)
#             q_sim = (q_g_normalized * q_normalized).sum(dim=-1).unsqueeze(-1)
#             q_sim[:,:5] = 0

#             k_normalized = k / k.norm(dim=-1, keepdim=True)
#             k_g_normalized = k_g / k_g.norm(dim=-1, keepdim=True)
#             k_sim = (k_g_normalized * k_normalized).sum(dim=-1).unsqueeze(-1)
#             k_sim[:,:5] = 0


#             q_augmented = q + (k * k_sim)
#             k_augmented = k + (q * q_sim)
#             v_augmented = v - ( v_g * (q_sim + k_sim))

#             x = memory_efficient_attention(q, k, v_augmented, attn_bias=attn_bias)
#         else:
#             new_g_info = None
#             x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x, new_g_info


"""
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
            g_info_layer = g_info[0]
            new_g_info = g_info[1:]

            g_info_layer = g_info_layer.reshape(
                g_info_layer.shape[0], self.num_heads, C // self.num_heads
            )
            q_g = g_info_layer[0].unsqueeze(0).unsqueeze(0)

            q_normalized = q / q.norm(dim=-1, keepdim=True)
            q_g_normalized = q_g / q_g.norm(dim=-1, keepdim=True)
            q_sim = (q_g_normalized * q_normalized).sum(dim=-1)
            q_weighted = q * q_sim.unsqueeze(-1)
            # q_weighted[:, :5] = 1
            k_augmented = k + q_weighted

            x = memory_efficient_attention(q, k_augmented, v, attn_bias=attn_bias)
        else:
            new_g_info = None
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, new_g_info"""
