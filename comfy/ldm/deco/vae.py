"""
PSVAE — Pixel-Semantic Visual Autoencoder for DeCo / Nanosaur.

Components:
  DINOv3 ViT-B/16 encoder (implemented from scratch, no timm dependency)
  SemanticEncoder  (projects DINO features to latent space)
  PixelDecoder     (standard convolutional decoder, reuses ComfyUI Decoder)
  PSVAE            (top-level wrapper with encode / decode interface)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.diffusionmodules.model import Decoder


# ---------------------------------------------------------------------------
# DINOv3 ViT-B/16  —  minimal re-implementation
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to patch embedding via a single strided convolution."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) → (B, N, embed_dim)
        x = self.proj(x)                       # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)       # (B, N, embed_dim)
        return x


class DINOMlp(nn.Module):
    """Two-layer MLP with GELU, matching timm's ``Mlp`` key names (fc1/fc2)."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class DINOAttention(nn.Module):
    """Multi-head self-attention with fused QKV (no QKV bias, as in DINOv3)."""

    def __init__(self, dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DINOBlock(nn.Module):
    """Transformer block with layer-scale (gamma_1 / gamma_2)."""

    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DINOMlp(dim, int(dim * mlp_ratio))
        # Layer-scale parameters
        self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.gamma_2 = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


class DINOEncoder(nn.Module):
    """
    Minimal DINOv3 ViT-B/16 encoder.

    State-dict keys produced (relative to this module)::

        patch_embed.proj.weight   [768, 3, 16, 16]
        patch_embed.proj.bias     [768]
        cls_token                 [1, 1, 768]
        reg_token                 [1, 4, 768]
        blocks.{0-11}.*           (12 DINOBlocks)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_blocks: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        in_chans: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 5  # 1 cls + 4 reg

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, 4, embed_dim))
        self.blocks = nn.ModuleList(
            [DINOBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  ImageNet-normalised image.

        Returns
        -------
        features : (B, N, embed_dim)  patch features (cls / reg tokens stripped).
        """
        B = x.shape[0]
        x = self.patch_embed(x)                              # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)
        reg = self.reg_token.expand(B, -1, -1)
        x = torch.cat([cls, reg, x], dim=1)                 # (B, 5+N, D)

        for blk in self.blocks:
            x = blk(x)

        # Strip cls + reg prefix tokens → pure patch features
        x = x[:, self.num_prefix_tokens:]                    # (B, N, D)
        return x


class DINOv3Encoder(nn.Module):
    """
    Wrapper that reproduces the nested attribute path used in the checkpoint::

        dino_encoder.model.model.<DINOEncoder params>
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, num_blocks: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        # nn.ModuleDict gives us the `.model.model.*` key path
        self.model = nn.ModuleDict({
            "model": DINOEncoder(embed_dim, num_heads, num_blocks)
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model["model"](x)


# ---------------------------------------------------------------------------
# Semantic encoder  —  ported directly from vae.py
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """LayerNorm → MultiheadAttention → LayerNorm → MLP."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SemanticEncoder(nn.Module):
    """Projects DINO features (768-d) to a compact latent (96-d)."""

    def __init__(self, in_dim: int = 768, latent_dim: int = 96, num_blocks: int = 3, num_heads: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(latent_dim, num_heads=num_heads) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.out_proj(x)
        return x


# ---------------------------------------------------------------------------
# Pixel decoder  —  wraps ComfyUI's standard convolutional Decoder
# ---------------------------------------------------------------------------

class PixelDecoder(nn.Module):
    """
    Wraps the standard Decoder from
    ``comfy.ldm.modules.diffusionmodules.model``.

    State-dict keys (relative to this module)::

        decoder.conv_in.*
        decoder.mid.*
        decoder.up.*
        decoder.norm_out.*
        decoder.conv_out.*
    """

    def __init__(self, latent_dim: int = 96, out_channels: int = 3, out_size: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = Decoder(
            ch=128,
            out_ch=out_channels,
            ch_mult=(1, 1, 2, 2, 4),
            num_res_blocks=2,
            attn_resolutions=[16],
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=out_channels,
            resolution=out_size,
            z_channels=latent_dim,
            tanh_out=False,
        )

    def forward(self, z: torch.Tensor, spatial_hw: tuple = None) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, N, C)  latent token sequence.
        spatial_hw : optional (h, w) to reshape tokens into a 2-D map.

        Returns
        -------
        image : (B, 3, H, W)  reconstructed image (ImageNet-normalised).
        """
        B, N, C = z.shape
        if spatial_hw is not None:
            h, w = spatial_hw
        else:
            h = w = int(N ** 0.5)
            assert h * w == N, f"N={N} is not a perfect square and spatial_hw not provided"

        x = z.permute(0, 2, 1).reshape(B, self.latent_dim, h, w)
        x = self.decoder(x)
        return x


# ---------------------------------------------------------------------------
# PSVAE  —  top-level model expected by ComfyUI's VAE class
# ---------------------------------------------------------------------------

class PSVAE(nn.Module):
    """
    Pixel-Semantic VAE used by DeCo / Nanosaur.

    Provides the ``encode`` / ``decode`` interface consumed by
    :class:`comfy.sd.VAE`.

    State-dict key prefixes::

        dino_encoder.model.model.*     DINOv3 ViT-B/16
        semantic_encoder.*             SemanticEncoder
        pixel_decoder.decoder.*        convolutional Decoder
    """

    def __init__(self, latent_dim: int = 96, dino_dim: int = 768):
        super().__init__()
        self.latent_dim = latent_dim
        self.dino_dim = dino_dim

        self.dino_encoder = DINOv3Encoder(embed_dim=dino_dim)
        self.semantic_encoder = SemanticEncoder(in_dim=dino_dim, latent_dim=latent_dim)
        self.pixel_decoder = PixelDecoder(latent_dim=latent_dim)

        # ImageNet normalisation constants (non-persistent → not in state_dict)
        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    # -- normalisation helpers ------------------------------------------------

    def _to_imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → ImageNet normalised."""
        x = (x + 1.0) * 0.5                      # → [0, 1]
        return (x - self.img_mean) / self.img_std

    def _from_imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet normalised → [-1, 1]."""
        x = x * self.img_std + self.img_mean      # → [0, 1]
        return x * 2.0 - 1.0

    # -- public interface -----------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image to a spatial latent map.

        Parameters
        ----------
        x : (B, 3, H, W) in [-1, 1]

        Returns
        -------
        z : (B, latent_dim, h, w)   where h = H/16, w = W/16
        """
        B, _, H, W = x.shape
        x_norm = self._to_imagenet_norm(x)
        dino_features = self.dino_encoder(x_norm)          # (B, N, 768)
        z = self.semantic_encoder(dino_features)            # (B, N, latent_dim)
        h, w = H // 16, W // 16
        z = z.permute(0, 2, 1).reshape(B, self.latent_dim, h, w)
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode a spatial latent map back to an image.

        Parameters
        ----------
        z : (B, latent_dim, h, w)

        Returns
        -------
        image : (B, 3, H, W)  in [-1, 1]
        """
        B, C, h, w = z.shape
        z_seq = z.reshape(B, C, h * w).permute(0, 2, 1)   # (B, h*w, C)
        x_norm = self.pixel_decoder(z_seq, spatial_hw=(h, w))
        return self._from_imagenet_norm(x_norm)
