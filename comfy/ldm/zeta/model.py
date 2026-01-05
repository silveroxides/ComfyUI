"""Zeta model for ComfyUI.

Main transformer model for Zeta, extending Lumina2/Z-Image architecture with DCT decoder.
Ported from Flow repository's model_dct.py.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import comfy.ldm.common_dit
import comfy.ldm.lumina.model as lumina  # For TimestepEmbedder only
import comfy.patcher_extension
from .layers import SimpleMLPAdaLN, RopeEmbedder, ZImageTransformerBlock, ZImageRMSNorm

# Diagnostic logging for Zeta model
_zeta_logger = None
_zeta_log_file = None

def _get_zeta_logger():
    """Get or create the Zeta diagnostic logger."""
    global _zeta_logger, _zeta_log_file
    if _zeta_logger is None:
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except:
            output_dir = "."
        _zeta_log_file = os.path.join(output_dir, f"zeta_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        _zeta_logger = logging.getLogger("ZetaDebug")
        _zeta_logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(_zeta_log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        _zeta_logger.addHandler(handler)
        _zeta_logger.info(f"Zeta diagnostic log started. Log file: {_zeta_log_file}")
    return _zeta_logger

def _log_zeta(msg: str):
    """Log a diagnostic message."""
    logger = _get_zeta_logger()
    logger.info(msg)


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

        # Log all received parameters for debugging
        _log_zeta("=" * 60)
        _log_zeta("Zeta.__init__ called with:")
        _log_zeta(f"  patch_size={patch_size}, in_channels={in_channels}")
        _log_zeta(f"  dim={dim}, n_layers={n_layers}, n_refiner_layers={n_refiner_layers}")
        _log_zeta(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}")
        _log_zeta(f"  axes_dims={axes_dims}, axes_lens={axes_lens}")
        _log_zeta(f"  rope_theta={rope_theta}, time_scale={time_scale}")
        _log_zeta(f"  cap_feat_dim={cap_feat_dim}, use_x0={use_x0}")
        _log_zeta(f"  decoder_hidden_size={decoder_hidden_size}, decoder_num_res_blocks={decoder_num_res_blocks}")
        _log_zeta(f"  ffn_dim_multiplier={ffn_dim_multiplier}, multiple_of={multiple_of}")
        _log_zeta(f"  z_image_modulation={z_image_modulation}, pad_tokens_multiple={pad_tokens_multiple}")
        _log_zeta(f"  image_model={image_model}")
        _log_zeta(f"  Extra kwargs: {kwargs}")
        _log_zeta(f"  Expected head_dim: dim/n_heads = {dim}/{n_heads} = {dim // n_heads}")
        _log_zeta(f"  Sum of axes_dims: {sum(axes_dims)}")
        _log_zeta(f"  head_dim == sum(axes_dims)? {dim // n_heads == sum(axes_dims)}")
        _log_zeta("=" * 60)

        # Input embedder
        self.x_embedder = operations.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        # Noise refiner blocks (use native Zeta blocks)
        adaln_embed_dim = 256
        self.noise_refiner = nn.ModuleList([
            ZImageTransformerBlock(
                1000 + layer_id,
                dim,
                n_heads,
                n_kv_heads,
                norm_eps,
                qk_norm,
                modulation=True,
                adaln_embed_dim=adaln_embed_dim,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            for layer_id in range(n_refiner_layers)
        ])

        # Context refiner blocks (use native Zeta blocks)
        self.context_refiner = nn.ModuleList([
            ZImageTransformerBlock(
                layer_id,
                dim,
                n_heads,
                n_kv_heads,
                norm_eps,
                qk_norm,
                modulation=False,
                adaln_embed_dim=adaln_embed_dim,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            for layer_id in range(n_refiner_layers)
        ])

        # Timestep embedder - checkpoint shows hidden_size=1024, output_size=256
        self.t_embedder = lumina.TimestepEmbedder(1024, frequency_embedding_size=256, output_size=adaln_embed_dim, dtype=dtype, device=device, operations=operations)

        # Caption embedder
        self.cap_embedder = nn.Sequential(
            operations.RMSNorm(cap_feat_dim, eps=norm_eps, device=device, dtype=dtype),
            operations.Linear(cap_feat_dim, dim, bias=True, device=device, dtype=dtype),
        )

        # Padding tokens
        self.x_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))

        # Main transformer layers (use native Zeta blocks)
        self.layers = nn.ModuleList([
            ZImageTransformerBlock(
                layer_id,
                dim,
                n_heads,
                n_kv_heads,
                norm_eps,
                qk_norm,
                modulation=True,
                adaln_embed_dim=adaln_embed_dim,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            for layer_id in range(n_layers)
        ])

        # RoPE embedder (use native Zeta RopeEmbedder)
        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)
        self.adaln_embed_dim = adaln_embed_dim

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
        
        # Log forward pass inputs
        _log_zeta("-" * 60)
        _log_zeta("Zeta._forward called:")
        _log_zeta(f"  Input x.shape: {x.shape} (B, C, H, W)")
        _log_zeta(f"  context.shape: {cap_feats.shape}")
        _log_zeta(f"  num_tokens: {num_tokens}")
        _log_zeta(f"  timesteps: {timesteps}")
        if attention_mask is not None:
            _log_zeta(f"  attention_mask.shape: {attention_mask.shape}")
        
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))

        # Store raw input for decoder
        B = x.shape[0]
        
        # Patchify: BCHW -> B, num_patches, patch_size^2 * C
        img = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        img = img.permute(0, 2, 3, 1, 4, 5).contiguous()
        num_patches_h, num_patches_w = img.shape[1], img.shape[2]
        num_patches = num_patches_h * num_patches_w
        img = img.view(B, num_patches, -1)  # [B, N, C*P*P]
        
        _log_zeta(f"  After patchify: num_patches_h={num_patches_h}, num_patches_w={num_patches_w}, total={num_patches}")
        _log_zeta(f"  img.shape after patchify: {img.shape}")
        
        # Store pixel values for decoder: [B*N, P*P, C]
        pixel_values = img.view(B * num_patches, self.patch_size ** 2, self.in_channels)

        # Timestep embedding
        t_emb = self.t_embedder(t * self.time_scale, dtype=x.dtype)
        adaln_input = t_emb

        # Embed inputs
        img_hidden = self.x_embedder(img)
        txt_hidden = self.cap_embedder(cap_feats)

        _log_zeta(f"  img_hidden.shape (after x_embedder): {img_hidden.shape}")
        _log_zeta(f"  txt_hidden.shape (after cap_embedder): {txt_hidden.shape}")
        _log_zeta(f"  t_emb.shape: {t_emb.shape}")

        # Compute position IDs and RoPE
        # For image: create position IDs based on patch positions
        img_ids = self._make_img_ids(num_patches_h, num_patches_w, num_tokens, x.device)
        txt_ids = self._make_txt_ids(num_tokens, x.device)
        
        _log_zeta(f"  img_ids.shape: {img_ids.shape}")
        _log_zeta(f"  txt_ids.shape: {txt_ids.shape}")
        _log_zeta(f"  img_ids expanded shape: {img_ids.unsqueeze(0).expand(B, -1, -1).shape}")
        _log_zeta(f"  txt_ids expanded shape: {txt_ids.unsqueeze(0).expand(B, -1, -1).shape}")
        
        img_pe = self.rope_embedder(img_ids.unsqueeze(0).expand(B, -1, -1))
        txt_pe = self.rope_embedder(txt_ids.unsqueeze(0).expand(B, -1, -1))

        _log_zeta(f"  img_pe.shape (after rope_embedder): {img_pe.shape}")
        _log_zeta(f"  txt_pe.shape (after rope_embedder): {txt_pe.shape}")
        _log_zeta(f"  Expected: complex tensor of shape (B, seq_len, head_dim/2) for native Zeta RoPE")

        # Masks
        img_mask = torch.ones((B, num_patches), device=x.device, dtype=torch.bool)
        if cap_mask is None:
            txt_mask = torch.ones((B, cap_feats.shape[1]), device=x.device, dtype=torch.bool)
        else:
            txt_mask = cap_mask.bool()

        _log_zeta(f"  img_mask.shape: {img_mask.shape}")
        _log_zeta(f"  txt_mask.shape: {txt_mask.shape}")
        _log_zeta("-" * 60)

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

    def _make_img_ids(self, h: int, w: int, txt_len: int, device) -> Tensor:
        """Create position IDs for image patches - matches Flow's format.
        
        Following Flow's prepare_latent_image_ids: creates (h, w, 3) grid where:
        - ids[..., 0] = offset (txt_len + 1, constant for all patches)
        - ids[..., 1] = row indices (0 to h-1)
        - ids[..., 2] = column indices (0 to w-1)
        Then flattened to (h*w, 3).
        """
        # Create grid like Flow does
        latent_image_ids = torch.zeros(h, w, 3, device=device)
        
        # First dim is offset (after text tokens)
        latent_image_ids[..., 0] = txt_len + 1
        
        # Second dim is row position
        latent_image_ids[..., 1] = torch.arange(h, device=device)[:, None].float()
        
        # Third dim is column position
        latent_image_ids[..., 2] = torch.arange(w, device=device)[None, :].float()
        
        # Flatten to (h*w, 3)
        return latent_image_ids.reshape(h * w, 3).long()

    def _make_txt_ids(self, length: int, device) -> Tensor:
        """Create position IDs for text tokens - matches Flow's format.
        
        Text IDs: each token gets increasing index in dim 0, zeros in dims 1,2.
        """
        ids = torch.zeros(length, 3, device=device, dtype=torch.long)
        ids[:, 0] = torch.arange(1, length + 1, device=device)
        return ids


    def _apply_x0_residual(self, predicted: Tensor, noisy: Tensor, timesteps: Tensor) -> Tensor:
        """Convert x0 prediction to v-prediction for flow matching."""
        eps = 0.0  # No epsilon needed at inference
        return (noisy - predicted) / (timesteps.view(-1, 1, 1, 1) + eps)
