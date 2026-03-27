from functools import lru_cache
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from comfy.ldm.modules.attention import optimized_attention


class Norm(nn.Module):
    """RMSNorm with learnable weight."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.w12 = operations.Linear(dim, hidden_dim * 2, bias=False, dtype=dtype, device=device)
        self.w3 = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Embed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 768, norm_layer=None, bias: bool = True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = operations.Linear(in_chans, embed_dim, bias=bias, dtype=dtype, device=device)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale=1.0):
    if isinstance(scale, (float, int)):
        scale = (float(scale), float(scale))
    scale_y, scale_x = float(scale[0]), float(scale[1])

    # 2D RoPE rotates pairs of channels, split evenly across x/y axes.
    rotary_dim = (dim // 4) * 4
    if rotary_dim == 0:
        return torch.empty(height * width, 0, 2, dtype=torch.float32)

    axis_dim = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))

    y_pos = (torch.arange(height, dtype=torch.float32) + 0.5) / height * scale_y
    x_pos = (torch.arange(width, dtype=torch.float32) + 0.5) / width * scale_x

    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)

    x_freqs = torch.outer(x_pos, inv_freq)
    y_freqs = torch.outer(y_pos, inv_freq)

    cos_x = torch.cos(x_freqs)
    sin_x = torch.sin(x_freqs)
    cos_y = torch.cos(y_freqs)
    sin_y = torch.sin(y_freqs)

    cos = torch.cat([cos_x, cos_y], dim=-1)
    sin = torch.cat([sin_x, sin_y], dim=-1)
    return torch.stack((cos, sin), dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs_cis.unbind(dim=-1)
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim == 0:
        return xq, xk

    cos = cos[None, None, :, :].to(dtype=xq.dtype, device=xq.device)
    sin = sin[None, None, :, :].to(dtype=xq.dtype, device=xq.device)

    xq_rot, xq_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    xk_rot, xk_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]

    xq1, xq2 = xq_rot.chunk(2, dim=-1)
    xk1, xk2 = xk_rot.chunk(2, dim=-1)
    xq_rot = torch.cat([xq1 * cos - xq2 * sin, xq1 * sin + xq2 * cos], dim=-1)
    xk_rot = torch.cat([xk1 * cos - xk2 * sin, xk1 * sin + xk2 * cos], dim=-1)

    if xq_pass.shape[-1] == 0:
        return xq_rot, xk_rot
    return torch.cat([xq_rot, xq_pass], dim=-1), torch.cat([xk_rot, xk_pass], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LocalContext2D(nn.Module):
    """
    2D Canon-style local mixing for DiT.
    Per-layer learned conv and scale.
    """
    def __init__(self, dim, num_layers, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList([
            operations.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ])
        self.lambdas = nn.Parameter(0.1 * torch.ones(num_layers))

    def forward(self, x, layer_idx, H, W):
        B, N, D = x.shape
        x_2d = x.view(B, H, W, D).permute(0, 3, 1, 2)
        local = self.convs[layer_idx](x_2d).permute(0, 2, 3, 1).view(B, N, D)
        return x + self.lambdas[layer_idx] * local


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            use_cross_attention: bool = True,
            dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_cross_attention = use_cross_attention
        self.qkv_x = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        if use_cross_attention:
            self.kv_y = operations.Linear(dim, dim * 2, bias=qkv_bias, dtype=dtype, device=device)

        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y, pos, y_token_weights: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv_x = self.qkv_x(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q)
        kx = self.k_norm(kx)
        q, kx = apply_rotary_emb(q, kx, freqs_cis=pos)

        if self.use_cross_attention:
            kv_y = self.kv_y(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            ky, vy = kv_y[0], kv_y[1]
            ky = self.k_norm(ky)

            k = torch.cat([kx, ky], dim=2)
            v = torch.cat([vx, vy], dim=2)
        else:
            k = kx
            v = vx

        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads)

        # Use ComfyUI optimized attention with skip_reshape=True
        # since q, k, v are already in (B, heads, seq_len, head_dim) format
        if self.use_cross_attention and y_token_weights is not None:
            y_token_weights = y_token_weights.to(device=q.device, dtype=q.dtype)
            y_token_bias = torch.log(torch.clamp(y_token_weights, min=1e-4))
            x_token_bias = torch.zeros(B, N, device=q.device, dtype=q.dtype)
            attn_bias = torch.cat([x_token_bias, y_token_bias], dim=1)[:, None, None, :]
            x = optimized_attention(q, k, v, self.num_heads, mask=attn_bias, skip_reshape=True)
        else:
            x = optimized_attention(q, k, v, self.num_heads, skip_reshape=True)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4, is_encoder_block=False, use_cross_attention=True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False, use_cross_attention=use_cross_attention,
                              dtype=dtype, device=device, operations=operations)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim, dtype=dtype, device=device, operations=operations)
        self.is_encoder_block = is_encoder_block
        if not is_encoder_block:
            self.adaLN_modulation = nn.Sequential(
                operations.Linear(hidden_size, 6 * hidden_size, bias=True, dtype=dtype, device=device)
            )

    def forward(
        self,
        x,
        y,
        c,
        pos,
        shared_adaLN=None,
        local_context=None,
        layer_idx=None,
        H=None,
        W=None,
        y_token_weights: torch.Tensor | None = None,
    ):
        if self.is_encoder_block:
            adaLN_output = shared_adaLN(c)
        else:
            adaLN_output = self.adaLN_modulation(c)

        if local_context is not None and H is not None and W is not None:
            x = local_context(x, layer_idx, H, W)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_output.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            y,
            pos,
            y_token_weights=y_token_weights,
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def precompute_freqs_cis_ex2d(dim: int, height: int, width: int, theta: float = 10000.0, scale=1.0):
    if isinstance(scale, float):
        scale = (scale, scale)

    x_pos = torch.linspace(0, height * scale[0], width)
    y_pos = torch.linspace(0, width * scale[1], height)

    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))  # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float()  # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float()  # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)  # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs=8,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            operations.Linear(in_channels + max_freqs ** 2, hidden_size_input, bias=True, dtype=dtype, device=device),
        )

    @lru_cache
    def fetch_pos(self, patch_size, device, dtype):
        pos = precompute_freqs_cis_ex2d(self.max_freqs ** 2 * 2, patch_size, patch_size)
        pos = pos[None, :, :].to(device=device, dtype=dtype)
        return pos

    @torch.compiler.disable
    def forward(self, inputs):
        B, P2, C = inputs.shape
        patch_size = int(P2 ** 0.5)
        device = inputs.device
        dtype = inputs.dtype
        dct = self.fetch_pos(patch_size, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs


class TextRefineAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv_x = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads)

        # Use ComfyUI optimized attention with skip_reshape=True
        x = optimized_attention(q, k, v, self.num_heads, skip_reshape=True)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TextRefineBlock(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = TextRefineAttention(hidden_size, num_heads=groups, qkv_bias=False,
                                        dtype=dtype, device=device, operations=operations)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim, dtype=dtype, device=device, operations=operations)

        self.adaLN_modulation = nn.Sequential(
            operations.Linear(hidden_size, 6 * hidden_size, bias=True, dtype=dtype, device=device)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.channels = channels

        self.in_ln = operations.LayerNorm(channels, eps=1e-6, dtype=dtype, device=device)
        self.mlp = nn.Sequential(
            operations.Linear(channels, channels, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(channels, channels, bias=True, dtype=dtype, device=device),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(channels, 3 * channels, bias=True, dtype=dtype, device=device)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(model_channels, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        patch_size,
        grad_checkpointing=False,
        dtype=None, device=None, operations=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.patch_size = patch_size

        self.cond_embed = operations.Linear(z_channels, patch_size ** 2 * model_channels, dtype=dtype, device=device)

        self.input_proj = operations.Linear(in_channels, model_channels, dtype=dtype, device=device)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                dtype=dtype, device=device, operations=operations,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels, dtype=dtype, device=device, operations=operations)

    def forward(self, x, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        c = self.cond_embed(c)

        y = c.reshape(c.shape[0], self.patch_size ** 2, -1)

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x)
