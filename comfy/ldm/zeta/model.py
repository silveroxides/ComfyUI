"""Zeta model for ComfyUI.

Main transformer model for Zeta, extending Lumina2/Z-Image architecture with DCT decoder.
Ported from Flow repository's model_dct.py.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import comfy.ldm.common_dit
import comfy.ldm.lumina.model as lumina
import comfy.patcher_extension
from .layers import SimpleMLPAdaLN


@dataclass
class ZetaParams:
    patch_size: int
    in_channels: int
    dim: int
    n_layers: int
    n_refiner_layers: int
    n_heads: int
    n_kv_heads: int
    multiple_of: int
    ffn_dim_multiplier: float
    norm_eps: float
    qk_norm: bool
    cap_feat_dim: int
    axes_dims: List[int]
    axes_lens: List[int]
    rope_theta: float
    time_scale: float
    decoder_hidden_size: int
    decoder_num_res_blocks: int
    decoder_max_freqs: int
    use_x0: bool
    z_image_modulation: bool
    pad_tokens_multiple: int

class Zeta(nn.Module):
    """
    Zeta: A diffusion transformer with DCT decoder for enhanced output.
    
    Key differences from base Lumina2/Z-Image:
    - in_channels: 128 (vs 16)
    - patch_size: 1 (vs 2) 
    - Includes dec_net (SimpleMLPAdaLN) for output decoding
    - Uses DCT positional encoding in decoder
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 8.0 / 3.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        axes_dims: List[int] = [32, 48, 48],
        axes_lens: List[int] = [1536, 512, 512],
        rope_theta: float = 256.0,
        time_scale: float = 1000.0,
        # DCT decoder params
        decoder_hidden_size: int = 3840,
        decoder_num_res_blocks: int = 4,
        decoder_max_freqs: int = 8,
        use_x0: bool = True,
        # Z-Image modulation flag (unused but passed from detection)
        z_image_modulation: bool = True,
        # Padding tokens multiple (optional)
        pad_tokens_multiple: int = None,
        image_model=None,
        device=None,
        dtype=None,
        operations=None,
        **kwargs,  # Capture any additional params from unet_config
    ) -> None:
        super().__init__()
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_scale = time_scale
        self.dim = dim
        self.n_heads = n_heads

        # Input embedder
        self.x_embedder = operations.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        # Noise refiner blocks  
        self.noise_refiner = nn.ModuleList([
            lumina.JointTransformerBlock(
                layer_id,
                dim,
                n_heads,
                n_kv_heads,
                multiple_of,
                ffn_dim_multiplier,
                norm_eps,
                qk_norm,
                modulation=True,
                z_image_modulation=True,
                operation_settings=operation_settings,
            )
            for layer_id in range(n_refiner_layers)
        ])

        # Context refiner blocks
        self.context_refiner = nn.ModuleList([
            lumina.JointTransformerBlock(
                layer_id,
                dim,
                n_heads,
                n_kv_heads,
                multiple_of,
                ffn_dim_multiplier,
                norm_eps,
                qk_norm,
                modulation=False,
                z_image_modulation=True,
                operation_settings=operation_settings,
            )
            for layer_id in range(n_refiner_layers)
        ])

        # Timestep embedder - checkpoint shows hidden_size=1024, output_size=256
        adaln_embed_dim = 256
        self.t_embedder = lumina.TimestepEmbedder(1024, frequency_embedding_size=256, output_size=adaln_embed_dim, dtype=dtype, device=device, operations=operations)

        # Caption embedder
        self.cap_embedder = nn.Sequential(
            operations.RMSNorm(cap_feat_dim, eps=norm_eps, device=device, dtype=dtype),
            operations.Linear(cap_feat_dim, dim, bias=True, device=device, dtype=dtype),
        )

        # Padding tokens
        self.x_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))

        # Main transformer layers
        self.layers = nn.ModuleList([
            lumina.JointTransformerBlock(
                layer_id,
                dim,
                n_heads,
                n_kv_heads,
                multiple_of,
                ffn_dim_multiplier,
                norm_eps,
                qk_norm,
                modulation=True,
                z_image_modulation=True,
                operation_settings=operation_settings,
            )
            for layer_id in range(n_layers)
        ])

        # RoPE embedder
        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = lumina.EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=axes_dims)

        # DCT decoder network
        self.dec_net = SimpleMLPAdaLN(
            in_channels=in_channels,
            model_channels=decoder_hidden_size,
            out_channels=in_channels,
            z_channels=dim,
            num_res_blocks=decoder_num_res_blocks,
            patch_size=patch_size,
            max_freqs=decoder_max_freqs,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # X0 variant support
        if use_x0:
            self.register_buffer("__x0__", torch.tensor([]))

        # Pixel space VAE marker - Zeta uses a placeholder VAE since it outputs
        # to pixel space internally via the DCT decoder
        self.register_buffer("pixel_space_vae", torch.tensor([]))

    def forward(self, x, timesteps, context, num_tokens, attention_mask=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, kwargs.get("transformer_options", {}))
        ).execute(x, timesteps, context, num_tokens, attention_mask, **kwargs)

    def _forward(self, x, timesteps, context, num_tokens, attention_mask=None, transformer_options={}, **kwargs):
        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape
        
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))

        # Store raw input for decoder
        B = x.shape[0]
        
        # Patchify: BCHW -> B, num_patches, patch_size^2 * C
        img = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        img = img.permute(0, 2, 3, 1, 4, 5).contiguous()
        num_patches_h, num_patches_w = img.shape[1], img.shape[2]
        num_patches = num_patches_h * num_patches_w
        img = img.view(B, num_patches, -1)  # [B, N, C*P*P]
        
        # Store pixel values for decoder: [B*N, P*P, C]
        pixel_values = img.view(B * num_patches, self.patch_size ** 2, self.in_channels)

        # Timestep embedding
        t_emb = self.t_embedder(t * self.time_scale, dtype=x.dtype)
        adaln_input = t_emb

        # Embed inputs
        img_hidden = self.x_embedder(img)
        txt_hidden = self.cap_embedder(cap_feats)

        # Compute position IDs and RoPE
        # For image: create position IDs based on patch positions
        img_ids = self._make_img_ids(num_patches_h, num_patches_w, num_tokens, x.device)
        txt_ids = self._make_txt_ids(num_tokens, x.device)
        
        img_pe = self.rope_embedder(img_ids.unsqueeze(0).expand(B, -1, -1))
        txt_pe = self.rope_embedder(txt_ids.unsqueeze(0).expand(B, -1, -1))

        # Masks
        img_mask = torch.ones((B, num_patches), device=x.device, dtype=torch.bool)
        if cap_mask is None:
            txt_mask = torch.ones((B, cap_feats.shape[1]), device=x.device, dtype=torch.bool)
        else:
            txt_mask = cap_mask.bool()

        # Noise refiner
        for layer in self.noise_refiner:
            img_hidden = layer(img_hidden, img_mask, img_pe, adaln_input)

        # Context refiner
        for layer in self.context_refiner:
            txt_hidden = layer(txt_hidden, txt_mask, txt_pe)

        # Concatenate for main transformer
        mixed_hidden = torch.cat((txt_hidden, img_hidden), dim=1)
        mixed_mask = torch.cat((txt_mask, img_mask), dim=1)
        mixed_pe = torch.cat((txt_pe, img_pe), dim=1)

        # Main transformer layers
        for layer in self.layers:
            mixed_hidden = layer(mixed_hidden, mixed_mask, mixed_pe, adaln_input)

        # Extract image hidden states
        img_hidden = mixed_hidden[:, txt_hidden.shape[1]:, ...]  # [B, N, dim]

        # Prepare for decoder: [B, N, dim] -> [B*N, dim]
        decoder_condition = img_hidden.reshape(B * num_patches, self.dim)

        # Pass through DCT decoder
        output = self.dec_net(pixel_values, decoder_condition)

        # Reshape output: [B*N, P*P, C] -> [B, N, C*P*P]
        output = output.reshape(B, num_patches, -1)

        # Unpatchify: B, num_patches, C*P*P -> B, C, H, W
        output = output.view(B, num_patches_h, num_patches_w, self.in_channels, self.patch_size, self.patch_size)
        output = output.permute(0, 3, 1, 4, 2, 5).contiguous()
        output = output.view(B, self.in_channels, num_patches_h * self.patch_size, num_patches_w * self.patch_size)

        # Crop to original size
        output = output[:, :, :h, :w]

        # Flip output (same as Z-Image)
        output = -output

        # If x0 variant, convert to v-prediction
        if hasattr(self, "__x0__"):
            return self._apply_x0_residual(output, x[:, :, :h, :w], timesteps)

        return output

    def _make_img_ids(self, h: int, w: int, offset: int, device) -> Tensor:
        """Create position IDs for image patches."""
        ids = torch.zeros(h * w, 3, device=device, dtype=torch.int32)
        ids[:, 0] = offset  # First dim is text offset
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                ids[idx, 1] = i
                ids[idx, 2] = j
        return ids

    def _make_txt_ids(self, length: int, device) -> Tensor:
        """Create position IDs for text tokens."""
        ids = torch.zeros(length, 3, device=device, dtype=torch.int32)
        ids[:, 0] = torch.arange(1, length + 1, device=device)
        return ids

    def _apply_x0_residual(self, predicted: Tensor, noisy: Tensor, timesteps: Tensor) -> Tensor:
        """Convert x0 prediction to v-prediction for flow matching."""
        eps = 0.0  # No epsilon needed at inference
        return (noisy - predicted) / (timesteps.view(-1, 1, 1, 1) + eps)
