import argparse
import torch
import torchvision.transforms
from torch import nn
from torchvision import transforms
import torch.nn.modules.utils as nn_utils
import math
import types
from pathlib import Path
from typing import Union, List, Tuple
from PIL import Image


# This code is modified from https://github.com/ShirAmir/dino-vit-features
class ViTExtractor:
    """This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(self, model: nn.Module = None, stride=(2, 2)):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model = ViTExtractor.patch_vit_resolution(model, stride)

        self._feats = {"token": [], "atten": [], "key": [], "query": [], "value": []}
        self.hook_handlers = []
        self.load_size = None

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(
            self, x: torch.Tensor, w: int, h: int
        ) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size[1]) // stride_hw[1]
            h0 = 1 + (h - patch_size[0]) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if stride == patch_size:  # nothing to do
            return model

        assert all(
            [
                (patch_size[i] // stride[i]) * stride[i] == patch_size[i]
                for i in range(len(stride))
            ]
        ), f"stride {stride} should divide patch_size {patch_size}"

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(
            ViTExtractor._fix_pos_enc(patch_size, stride), model
        )
        return model

    def _get_hook_qkv(self):
        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = (
                module.qkv(input)
                .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            self._feats["query"].append(qkv[0])
            self._feats["key"].append(qkv[1])
            self._feats["value"].append(qkv[2])

        return _inner_hook

    def _get_hook_token_atten(self, facet: str):
        def _inner_hook(module, input, output):
            self._feats[facet].append(output)

        return _inner_hook

    def _register_hooks(self, layer_idx: list) -> None:
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layer_idx:
                self.hook_handlers.append(
                    block.register_forward_hook(self._get_hook_token_atten("token"))
                )
                self.hook_handlers.append(
                    block.attn.attn_drop.register_forward_hook(
                        self._get_hook_token_atten("atten")
                    )
                )
                self.hook_handlers.append(
                    block.attn.register_forward_hook(self._get_hook_qkv())
                )

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(
        self, batch: torch.Tensor, g_info: torch.Tensor = None, layer_idx: list = [11]
    ) -> List[torch.Tensor]:
        B, C, H, W = batch.shape
        self._feats = {"token": [], "atten": [], "key": [], "query": [], "value": []}
        self._register_hooks(layer_idx)
        cls_token = self.model(batch, g_info=g_info)
        self._unregister_hooks()
        self.load_size = (H, W)
        self._feats["cls_token"] = cls_token.unsqueeze(1)
        return self._feats

    def extract_descriptors(
        self,
        batch: torch.Tensor,
        g_info: torch.Tensor = None,
        layer_idx: list = [23],
        register_size=4,
    ) -> torch.Tensor:
        self._extract_features(batch, g_info, layer_idx)
        # equalize the dimensions of the cls token and the rest of the tokens
        self._feats["cls_token"] = [
            self._feats["cls_token"] for item in self._feats["token"]
        ]
        
        self._feats["token"] = [
            item[0][:, 1 + register_size :, :] for item in self._feats["token"]
        ] 
        # self._feats["atten"] = self._feats["atten"][0][:, 1 + register_size :, :]

        self._feats["key"] = [
            item.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)[
                :, 1 + register_size :, :
            ]
            for item in self._feats["key"]
        ]

        self._feats["query"] = [
            item.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)[
                :, 1 + register_size :, :
            ]
            for item in self._feats["query"]
        ]
        self._feats["value"] = [
            item.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)[
                :, 1 + register_size :, :
            ]
            for item in self._feats["value"]
        ]

        self._feats["token"] = torch.stack(self._feats["token"], dim=0).permute(1, 0, 2, 3)
        self._feats["key"] = torch.stack(self._feats["key"], dim=0).permute(1, 0, 2, 3)
        self._feats["query"] = torch.stack(self._feats["query"], dim=0).permute(1, 0, 2, 3)
        self._feats["value"] = torch.stack(self._feats["value"], dim=0).permute(1, 0, 2, 3)
        self._feats["cls_token"] = torch.stack(self._feats["cls_token"], dim=0).permute(1, 0, 2, 3)

        return self._feats

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert (
            self.model_type == "dino_vits8"
        ), f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], "attn")
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0]  # Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1)  # Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (
            temp_maxs - temp_mins
        )  # normalize to range [0,1]
        return cls_attn_maps
