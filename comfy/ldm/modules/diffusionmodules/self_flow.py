"""
Self-Flow Model.

This module contains the SelfFlowPerTokenDiT model, a Diffusion Transformer
with per-token timestep conditioning for Self-Flow training.
Adapted for ComfyUI.
"""

import collections.abc
import math
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import comfy.ops

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchedPatchEmbed(nn.Module):
    """Simplified Sequence to Patch Embedding using Linear layer."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
        operations=None
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        img_size = to_2tuple(img_size)

        self.grid_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        patch_dim = self.patch_size[0] * self.patch_size[1] * in_chans
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.proj = operations.Linear(patch_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


def modulate(x, shift, scale):
    """Standard modulation with unsqueeze for (N, D) conditioning."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def modulate_per_token(x, shift, scale):
    """Per-token modulation for (N, T, D) conditioning."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256, operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""

    def __init__(self, num_classes, hidden_size, dropout_prob, operations=None):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            operations=None
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        if operations is None:
            operations = comfy.ops.disable_weight_init

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = operations.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        import comfy.ldm.modules.attention as attention
        x = attention.optimized_attention(q, k, v, self.num_heads)
        
        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., operations=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.fc1 = operations.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = operations.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, operations=None, **block_kwargs):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, operations=operations, **block_kwargs
        )
        self.norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
            operations=operations
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), operations.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class PerTokenDiTBlock(DiTBlock):
    """DiT block that handles per-token conditioning (N, T, D) instead of (N, D)."""

    def forward(self, x, c):
        """
        Args:
            x: (N, T, D) tokens
            c: (N, T, D) per-token conditioning
        """
        batch_size, seq_len, hidden_dim = c.shape
        c_flat = c.reshape(-1, hidden_dim)
        modulation_flat = self.adaLN_modulation(c_flat)
        modulation = modulation_flat.reshape(batch_size, seq_len, -1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        x = x + gate_msa * self.attn(modulate_per_token(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate_per_token(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, patch_size, out_channels, operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = operations.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PerTokenFinalLayer(FinalLayer):
    """Final layer that handles per-token conditioning (N, T, D) instead of (N, D)."""

    def forward(self, x, c):
        """
        Args:
            x: (N, T, D) tokens
            c: (N, T, D) per-token conditioning
        """
        batch_size, seq_len, hidden_dim = c.shape
        c_flat = c.reshape(-1, hidden_dim)
        modulation_flat = self.adaLN_modulation(c_flat)
        modulation = modulation_flat.reshape(batch_size, seq_len, -1)

        shift, scale = modulation.chunk(2, dim=-1)
        x = modulate_per_token(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleHead(nn.Module):
    """Simple projection head for self-distillation."""
    def __init__(self, in_dim, out_dim, operations=None):
        super(SimpleHead, self).__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.linear1 = operations.Linear(in_dim, in_dim + out_dim)
        self.linear2 = operations.Linear(in_dim + out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(self.act(x))
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def batched_prc_img(x):
    """Convert (B, C*P*P, H/P, W/P) -> (B, L, C*P*P) with identity ids."""
    batch_size, channels, height, width = x.shape
    x_flat = rearrange(x, "b c h w -> b (h w) c")
    seq_len = x_flat.shape[1]
    
    # Create ids that match the sequence length
    x_ids = torch.zeros(batch_size, seq_len, 3, dtype=torch.float32, device=x.device)
    
    # We construct x_ids in a format similar to what other models expect for continuous coordinates
    grid_h = torch.arange(height, dtype=torch.float32, device=x.device)
    grid_w = torch.arange(width, dtype=torch.float32, device=x.device)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')
    
    # Flatten grids to match sequence length
    grid_h = grid_h.reshape(-1)
    grid_w = grid_w.reshape(-1)
    
    # Store normalized coordinates
    x_ids[:, :, 1] = grid_h.unsqueeze(0).expand(batch_size, -1) / height
    x_ids[:, :, 2] = grid_w.unsqueeze(0).expand(batch_size, -1) / width
    
    return x_flat, x_ids

class SelfFlowDiT(nn.Module):
    """
    Base Self-Flow DiT model.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
        learn_sigma=False,
        compatibility_mode=False,
        operations=None
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.compatibility_mode = compatibility_mode

        self.x_embedder = PatchedPatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True, operations=operations
        )
        self.t_embedder = TimestepEmbedder(hidden_size, operations=operations)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob=0.0, operations=operations)
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, operations=operations) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, operations=operations)
        
        # Self-distillation projector
        self.projector = SimpleHead(hidden_size, hidden_size, operations=operations)
        
    def shufflechannel(self, x):
        """Reorder channels/patches to match expected output format."""
        p = self.x_embedder.patch_size[0]
        x = rearrange(x, "b l (p q c) -> b l (c p q)", p=p, q=p, c=self.out_channels)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=2)
        return x

    def _forward(self, x, t, y, return_features=False, return_raw_features=False):
        """forward pass."""
        assert not (return_raw_features and return_features)
        
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if (i + 1) == return_features:
                zs = self.projector(x)
            elif (i + 1) == return_raw_features:
                zs = x
        
        x = self.final_layer(x, c)
        
        if return_features or return_raw_features:
            return x, zs
        else:
            return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        **kwargs
    ):
        """Forward pass adapted for ComfyUI."""
        # Convert latents from (B, C, H, W) to patched tokens
        batch_size = x.shape[0]
        # x is expected to be B, C, H, W. We need to patchify it.
        # Patchify noise: (B, C, H, W) -> (B, C*P*P, H/P, W/P)
        p1 = p2 = self.patch_size
        x_patched = rearrange(
            x,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=p1, p2=p2
        )
        
        x_tokens, x_ids = batched_prc_img(x_patched)
        
        # In ComfyUI, timesteps are passed directly. 
        # In the original repo, sample.py did `timesteps = 1 - timesteps`.
        # This will be handled in the ModelSampling class instead.
        
        # Context is usually the text embeddings, but SelfFlow uses class labels.
        # ComfyUI passes (B, N, D) context. We will take the first token's projection or use kwargs.
        # Let's assume context is passed appropriately or we fallback to 0.
        if "vector" in kwargs:
            y = kwargs["vector"]
        else:
            # dummy class labels for unconditional
            y = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            
        out = self._forward(x=x_tokens, t=timesteps, y=y)
        out = self.shufflechannel(out)
        
        # Unpatchify back to (B, C, H, W)
        out_unpatched = rearrange(
            out,
            "b (h w) (c p1 p2) -> b (c p1 p2) h w",
            h=x.shape[2]//p1, w=x.shape[3]//p2, p1=p1, p2=p2
        )
        
        out_img = rearrange(
            out_unpatched,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=p1, p2=p2, c=x.shape[1]
        )
        
        return -out_img


class SelfFlowPerTokenDiT(SelfFlowDiT):
    """
    Self-Flow DiT with per-token timestep conditioning.
    """

    def __init__(self, **kwargs):
        # Initialize parent class (creates standard DiTBlocks)
        super().__init__(**kwargs)

        hidden_size = kwargs.get("hidden_size", 1152)
        num_heads = kwargs.get("num_heads", 16)
        mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        patch_size = kwargs.get("patch_size", 2)
        in_channels = kwargs.get("in_channels", 4)
        learn_sigma = kwargs.get("learn_sigma", True)
        operations = kwargs.get("operations", None)

        out_channels = in_channels * 2 if learn_sigma else in_channels

        # Convert standard blocks to per-token versions, preserving weights
        self._convert_to_per_token_blocks(hidden_size, num_heads, mlp_ratio, patch_size, out_channels, operations)

    def _convert_to_per_token_blocks(self, hidden_size, num_heads, mlp_ratio, patch_size, out_channels, operations):
        """Convert DiTBlocks to PerTokenDiTBlocks while preserving weights."""
        new_blocks = nn.ModuleList()
        for original_block in self.blocks:
            new_block = PerTokenDiTBlock(hidden_size, num_heads, mlp_ratio, operations=operations)
            # Avoid loading state dict here if not initialized, handled during checkpoint loading
            new_blocks.append(new_block)
        self.blocks = new_blocks

        new_final = PerTokenFinalLayer(hidden_size, patch_size, out_channels, operations=operations)
        self.final_layer = new_final

    def _forward(self, x, t, y, return_features=False, return_raw_features=False):
        """Forward with per-token timestep conditioning."""
        assert not (return_raw_features and return_features)

        x = self.x_embedder(x) + self.pos_embed
        batch_size, seq_len, hidden_dim = x.shape

        # Handle timestep embedding - per-token or broadcast
        if t.ndim == 1:
            t_emb = self.t_embedder(t).unsqueeze(1).expand(-1, seq_len, -1)
        elif t.ndim == 2:
            t_flat = t.reshape(-1)
            t_emb_flat = self.t_embedder(t_flat)
            t_emb = t_emb_flat.reshape(batch_size, seq_len, -1)
        else:
            raise ValueError(f"Timesteps must be 1D or 2D, got shape {t.shape}")

        # Class embedding (broadcast to per-token)
        y_emb = self.y_embedder(y, self.training).unsqueeze(1).expand(-1, seq_len, -1)

        # Combine embeddings
        c = t_emb + y_emb

        # Apply per-token blocks
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if (i + 1) == return_features:
                zs = self.projector(x)
            elif (i + 1) == return_raw_features:
                zs = x

        # Apply per-token final layer
        x = self.final_layer(x, c)

        if return_features or return_raw_features:
            return x, zs
        else:
            return x
