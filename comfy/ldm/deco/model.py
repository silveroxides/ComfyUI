from dataclasses import dataclass

import math
from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange

_compile_disable = torch.compiler.disable if hasattr(torch, "compiler") else lambda f: f

from .layers import (
    Embed,
    NerfEmbedder,
    TimestepEmbedder,
    FlattenDiTBlock,
    TextRefineBlock,
    SimpleMLPAdaLN,
    precompute_freqs_cis_2d,
    LocalContext2D,
)


@dataclass
class DeCoParams:
    in_channels: int = 4
    num_groups: int = 12
    hidden_size: int = 1152
    decoder_hidden_size: int = 64
    num_encoder_blocks: int = 18
    num_decoder_blocks: int = 4
    num_text_blocks: int = 4
    patch_size: int = 2
    txt_embed_dim: int = 1024
    txt_max_length: int = 100
    alignment_layer: int = 8
    # SPRINT parameters
    sprint_num_f: int = 2
    sprint_num_h: int = 2
    sprint_drop_ratio: float = 0.75
    sprint_path_drop_prob: float = 0.05
    sprint_noise_std: float = 0
    rope_scale: float = 2 * math.pi
    # Experiment flags
    experiment: str = "local_context"


class DeCo(nn.Module):
    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        params = DeCoParams(**kwargs)
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.num_groups = params.num_groups
        self.decoder_hidden_size = params.decoder_hidden_size
        self.num_encoder_blocks = params.num_encoder_blocks
        self.num_decoder_blocks = params.num_decoder_blocks
        self.num_text_blocks = params.num_text_blocks
        self.patch_size = params.patch_size
        self.txt_embed_dim = params.txt_embed_dim
        self.txt_max_length = params.txt_max_length
        self.alignment_layer = params.alignment_layer
        # SPRINT parameters
        self.sprint_num_f = params.sprint_num_f
        self.sprint_num_h = params.sprint_num_h
        self.sprint_num_g = self.num_encoder_blocks - self.sprint_num_f - self.sprint_num_h
        self.sprint_drop_ratio = params.sprint_drop_ratio
        self.sprint_path_drop_prob = params.sprint_path_drop_prob
        self.sprint_noise_std = params.sprint_noise_std
        self.rope_scale = params.rope_scale
        # Experiment flags
        self.experiment = params.experiment

        assert (self.hidden_size // self.num_groups) % 4 == 0, \
            f"head_dim ({self.hidden_size // self.num_groups}) must be divisible by 4 for 2D RoPE"

        assert self.sprint_num_g >= 0, f"num_encoder_blocks ({self.num_encoder_blocks}) too small for SPRINT split (num_f={self.sprint_num_f}, num_h={self.sprint_num_h})"

        # SPRINT modules: mask token and fusion projection
        if self.sprint_drop_ratio > 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        else:
            self.register_buffer('mask_token', torch.zeros(1, 1, self.hidden_size))
        self.fusion_proj = operations.Linear(2 * self.hidden_size, self.hidden_size, bias=True, dtype=dtype, device=device)

        self.s_embedder = Embed(self.in_channels * self.patch_size ** 2, self.hidden_size, bias=True,
                                dtype=dtype, device=device, operations=operations)
        self.x_embedder = NerfEmbedder(self.in_channels, self.decoder_hidden_size, max_freqs=8,
                                       dtype=dtype, device=device, operations=operations)
        self.t_embedder = TimestepEmbedder(self.hidden_size,
                                           dtype=dtype, device=device, operations=operations)
        self.y_embedder = Embed(self.txt_embed_dim, self.hidden_size, bias=True, norm_layer=operations.RMSNorm,
                                dtype=dtype, device=device, operations=operations)

        self.shared_encoder_adaLN = nn.Sequential(
            operations.Linear(self.hidden_size, 6 * self.hidden_size, bias=True, dtype=dtype, device=device)
        )

        encoder_blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups, is_encoder_block=True,
                            use_cross_attention=(i % 2 == 0),
                            dtype=dtype, device=device, operations=operations)
            for i in range(self.num_encoder_blocks)
        ])
        self.blocks = nn.ModuleList(encoder_blocks)
        self.text_refine_blocks = nn.ModuleList([
            TextRefineBlock(self.hidden_size, self.num_groups,
                            dtype=dtype, device=device, operations=operations)
            for _ in range(self.num_text_blocks)
        ])

        self.local_context = None
        if "local_context" in self.experiment:
            self.local_context = LocalContext2D(self.hidden_size, self.num_encoder_blocks,
                                               dtype=dtype, device=device, operations=operations)

        self.dec_net = SimpleMLPAdaLN(
            in_channels=self.decoder_hidden_size,
            model_channels=self.decoder_hidden_size,
            out_channels=self.in_channels,
            z_channels=self.hidden_size,
            num_res_blocks=self.num_decoder_blocks,
            patch_size=self.patch_size,
            grad_checkpointing=False,
            dtype=dtype, device=device, operations=operations,
        )

        self.precompute_pos = dict()

    @_compile_disable
    def fetch_pos(self, height, width, device):
        height = int(height)
        width = int(width)
        key = (height, width)
        pos = self.precompute_pos.get(key)
        if pos is None:
            pos = precompute_freqs_cis_2d(
                self.hidden_size // self.num_groups,
                height,
                width,
                scale=self.rope_scale,
            )
            self.precompute_pos[key] = pos
        return pos.to(device=device)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        transformer_options={},
        **kwargs,
    ) -> torch.Tensor:
        t = timestep
        cond = context
        uncond = transformer_options.get("uncond", False)

        device = self.s_embedder.proj.weight.device
        embed_dtype = self.s_embedder.proj.weight.dtype
        if cond.device != device:
            cond = cond.to(device)

        y_emb = self.y_embedder(cond)
        y_emb = y_emb.view(cond.size(0), -1, self.hidden_size).to(embed_dtype)

        B, _, H, W = x.shape
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)

        xpos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)

        t = self.t_embedder(t.view(-1), dtype=embed_dtype).view(B, -1, self.hidden_size)
        condition = torch.nn.functional.silu(t)
        y_emb = y_emb.to(dtype=t.dtype)

        y_latent = y_emb
        for block in self.text_refine_blocks:
            y_latent = block(y_latent, condition)

        s = self.s_embedder(x)
        T = s.shape[1]
        H_patches, W_patches = H // self.patch_size, W // self.patch_size

        # 1) Dense encoder blocks
        for i in range(self.sprint_num_f):
            s = self.blocks[i](
                s, y_latent, condition, xpos,
                shared_adaLN=self.shared_encoder_adaLN,
                local_context=self.local_context,
                layer_idx=i, H=H_patches, W=W_patches,
            )

        s_enc = s

        # 2) Middle blocks (no token drop at inference)
        s_sparse = s
        for i in range(self.sprint_num_f, self.sprint_num_f + self.sprint_num_g):
            if not uncond:
                s_sparse = self.blocks[i](
                    s_sparse, y_latent, condition, xpos,
                    shared_adaLN=self.shared_encoder_adaLN,
                )

        # 3) Path-drop: replace sparse path with mask token for unconditional
        g_pad = s_sparse
        if uncond:
            g_pad = self.mask_token.expand_as(g_pad)

        # 4) Sparse-dense fusion
        s = self._sprint_fuse(s_enc, g_pad)

        # 5) Final encoder blocks
        for i in range(self.sprint_num_f + self.sprint_num_g, self.num_encoder_blocks):
            s = self.blocks[i](
                s, y_latent, condition, xpos,
                shared_adaLN=self.shared_encoder_adaLN,
                local_context=self.local_context,
                layer_idx=i, H=H_patches, W=W_patches,
            )

        s = torch.nn.functional.silu(t + s)
        batch_size, length, _ = s.shape

        x = x.reshape(batch_size * length, self.in_channels, self.patch_size ** 2)
        x = x.transpose(1, 2)
        s = s.view(batch_size * length, self.hidden_size)
        x = self.x_embedder(x)

        x = self.dec_net(x, s)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, length, -1)
        x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                       h=H // self.patch_size, w=W // self.patch_size,
                       p1=self.patch_size, p2=self.patch_size, c=self.in_channels)

        return x

    def _pad_with_mask(self, x_sparse, ids_keep, T_full):
        """
        Pad sparse tokens back to full length with mask tokens.

        Args:
            x_sparse: (B, T_keep, C) sparse tokens
            ids_keep: (T_keep,) indices of kept tokens (same for all batches)
            T_full: full sequence length T
        Returns:
            x_pad: (B, T_full, C) with mask tokens at dropped positions
        """
        if ids_keep is None:
            return x_sparse

        B, T_keep, C = x_sparse.shape
        assert T_full >= T_keep
        x_pad = self.mask_token.to(x_sparse.dtype).expand(B, T_full, C).clone()
        # Expand ids_keep for scatter: (T_keep,) -> (B, T_keep, C)
        ids_expanded = ids_keep.view(1, -1, 1).expand(B, -1, C)
        x_pad.scatter_(1, ids_expanded, x_sparse)
        return x_pad

    def _sprint_fuse(self, f_dense, g_pad):
        """
        Sparse-dense residual fusion.

        Args:
            f_dense: (B, T, C) encoder output (dense)
            g_pad: (B, T, C) padded sparse output
        Returns:
            h: (B, T, C) fused representation
        """
        h = torch.cat([f_dense, g_pad], dim=-1)  # (B, T, 2C)
        h = self.fusion_proj(h)
        return h
