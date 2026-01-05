"""ZetaDCT model layers for ComfyUI.

Layer components for the ZetaDCT (DCT decoder variant) model, ported from Flow repository.
"""

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply AdaLN modulation."""
    return x * (1 + scale) + shift


class NerfEmbedder(nn.Module):
    """
    An embedder module that combines input features with a 2D positional
    encoding that mimics the Discrete Cosine Transform (DCT).

    Takes an input tensor of shape (B, P^2, C), where P is the patch size,
    and enriches it with positional information before projecting to hidden size.
    """

    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input

        self.embedder = nn.Sequential(
            operations.Linear(in_channels + max_freqs ** 2, hidden_size_input, device=device, dtype=dtype)
        )

    @lru_cache(maxsize=4)
    def fetch_pos(self, patch_size: int, device, dtype):
        """Generates and caches 2D DCT-like positional embeddings."""
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")

        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]

        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)

        return dct

    def forward(self, inputs: Tensor) -> Tensor:
        B, P2, C = inputs.shape
        original_dtype = inputs.dtype

        with torch.autocast("cuda", enabled=False):
            patch_size = int(P2 ** 0.5)
            inputs_float = inputs.float()
            dct = self.fetch_pos(patch_size, inputs.device, torch.float32)
            dct = dct.repeat(B, 1, 1)
            inputs_cat = torch.cat([inputs_float, dct], dim=-1)
            # Run embedder in float32
            embedder_float = self.embedder[0].float()
            out = embedder_float(inputs_cat)

        return out.to(original_dtype)


class ResBlock(nn.Module):
    """
    A residual block with AdaLN modulation.
    Initialized to identity (zero modulation output).
    """

    def __init__(self, channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.channels = channels

        self.in_ln = operations.LayerNorm(channels, eps=1e-6, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            operations.Linear(channels, channels, bias=True, device=device, dtype=dtype),
            nn.SiLU(),
            operations.Linear(channels, channels, bias=True, device=device, dtype=dtype),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(channels, 3 * channels, bias=True, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class DCTFinalLayer(nn.Module):
    """The final layer adopted from DiT."""

    def __init__(self, model_channels: int, out_channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.linear = operations.Linear(model_channels, out_channels, bias=True, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP decoder for ZetaDCT variant.
    Uses NerfEmbedder for input projection and ResBlocks with AdaLN.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        patch_size: int,
        max_freqs: int = 8,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.patch_size = patch_size

        # Condition embedding from transformer backbone
        self.cond_embed = operations.Linear(z_channels, patch_size ** 2 * model_channels, device=device, dtype=dtype)

        # NerfEmbedder for input projection with DCT positional encoding
        self.input_embedder = NerfEmbedder(
            in_channels=in_channels,
            hidden_size_input=model_channels,
            max_freqs=max_freqs,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # Residual blocks with AdaLN
        self.res_blocks = nn.ModuleList([
            ResBlock(model_channels, dtype=dtype, device=device, operations=operations)
            for _ in range(num_res_blocks)
        ])

        # Final layer
        self.final_layer = DCTFinalLayer(model_channels, out_channels, dtype=dtype, device=device, operations=operations)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Apply the model to an input batch.
        :param x: an [N x P^2 x C] Tensor of inputs (pixel values per patch).
        :param c: conditioning from transformer backbone [N x z_channels].
        :return: an [N x P^2 x C] Tensor of outputs.
        """
        # Project input with DCT positional encoding
        x = self.input_embedder(x)

        # Embed condition and reshape for per-position modulation
        c = self.cond_embed(c)
        y = c.reshape(c.shape[0], self.patch_size ** 2, -1)

        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x)
