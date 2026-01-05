"""Zeta model layers for ComfyUI.

Layer components for the Zeta (DCT decoder variant) model, ported from Flow repository.
Includes transformer components (RopeEmbedder, ZImageAttention, ZImageTransformerBlock) 
and decoder components (NerfEmbedder, SimpleMLPAdaLN).
"""

import math
from functools import lru_cache
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# RoPE (Rotary Positional Embedding) Components
# ============================================================================

def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor.
    
    Args:
        x_in: Input tensor of shape [..., n_heads, head_dim]
        freqs_cis: Complex frequency tensor from RopeEmbedder
    
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class RopeEmbedder:
    """Rotary Position Embedding generator for Zeta model.
    
    Unlike Lumina2's EmbedND, this is a simple class (not nn.Module) that
    precomputes and caches frequency tensors for RoPE.
    """
    
    def __init__(
        self,
        theta: float = 256,
        axes_dims: List[int] = [32, 48, 48],
        axes_lens: List[int] = [1536, 512, 512],
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256):
        """Precompute complex frequency tensors for each axis."""
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (
                    theta
                    ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
                )
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(
                    torch.complex64
                )
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        """Generate RoPE frequencies for given position IDs.
        
        Args:
            ids: Position IDs of shape [..., 3] where last dim is (text_offset, h, w)
        
        Returns:
            Complex frequency tensor for RoPE
        """
        assert ids.ndim >= 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(
                self.axes_dims, self.axes_lens, theta=self.theta
            )
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[..., i].long()
            result.append(self.freqs_cis[i][index])

        return torch.cat(result, dim=-1)


# ============================================================================
# Transformer Components
# ============================================================================

class ZImageRMSNorm(nn.Module):
    """RMS Normalization layer for Zeta."""
    
    def __init__(self, dim: int, eps: float = 1e-5, dtype=None, device=None, operations=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class ZImageFeedForward(nn.Module):
    """Feed-forward network for Zeta transformer blocks."""
    
    def __init__(self, dim: int, hidden_dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.w1 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageAttention(nn.Module):
    """Multi-head attention for Zeta model with RoPE support."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.to_q = operations.Linear(dim, n_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.to_k = operations.Linear(dim, n_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.to_v = operations.Linear(dim, n_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.to_out = operations.Linear(n_heads * self.head_dim, dim, bias=False, device=device, dtype=dtype)

        if qk_norm:
            self.norm_q = ZImageRMSNorm(self.head_dim, eps=eps, device=device, dtype=dtype, operations=operations)
            self.norm_k = ZImageRMSNorm(self.head_dim, eps=eps, device=device, dtype=dtype, operations=operations)
        else:
            self.norm_q = None
            self.norm_k = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.n_heads, -1))
        key = key.unflatten(-1, (self.n_kv_heads, -1))
        value = value.unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Process attention mask
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
            if attention_mask.dtype == torch.bool:
                new_mask = torch.zeros_like(attention_mask, dtype=dtype)
                new_mask.masked_fill_(~attention_mask, float("-inf"))
                attention_mask = new_mask

        # Transpose for SDPA: (B, seq, heads, head_dim) -> (B, heads, seq, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Transpose back and flatten heads
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = self.to_out(hidden_states)
        return output


class ZImageTransformerBlock(nn.Module):
    """Transformer block for Zeta model with Z-Image style modulation."""
    
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        adaln_embed_dim: int = 256,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps, 
                                          dtype=dtype, device=device, operations=operations)
        hidden_dim = int(dim / 3 * 8)
        self.feed_forward = ZImageFeedForward(dim=dim, hidden_dim=hidden_dim, 
                                               dtype=dtype, device=device, operations=operations)

        self.attention_norm1 = ZImageRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.ffn_norm1 = ZImageRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.attention_norm2 = ZImageRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.ffn_norm2 = ZImageRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)

        if modulation:
            self.adaLN_modulation = operations.Linear(min(dim, adaln_embed_dim), 4 * dim, bias=True, 
                                                       device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            )
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            )
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


# ============================================================================
# Decoder Components
# ============================================================================

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
