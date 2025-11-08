# Credits:
# Original Flux code can be found on: https://github.com/black-forest-labs/flux
# Chroma Radiance adaption referenced from https://github.com/lodestone-rock/flow

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn
from einops import repeat
import comfy.ldm.common_dit

from comfy.ldm.flux.layers import EmbedND

from comfy.ldm.chroma.model import Chroma, ChromaParams
from comfy.ldm.chroma.layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
    Approximator,
)
from .layers import (
    NerfEmbedder,
    NerfGLUBlock,
    NerfFinalLayer,
    NerfFinalLayerConv,
)


@dataclass
class ChromaRadianceParams(ChromaParams):
    patch_size: int
    nerf_hidden_size: int
    nerf_mlp_ratio: int
    nerf_depth: int
    nerf_max_freqs: int
    nerf_tile_size: int
    nerf_final_head_type: str
    nerf_embedder_dtype: Optional[torch.dtype]
    grid_mitigation_enabled: bool = field(default=False)
    num_offsets: int = field(default=1)
    offset_size: int = field(default=15)


class ChromaRadiance(Chroma):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        if operations is None:
            raise RuntimeError("Attempt to create ChromaRadiance object without setting operations")
        nn.Module.__init__(self)
        self.dtype = dtype
        kwargs.setdefault('grid_mitigation_enabled', False)
        kwargs.setdefault('num_offsets', 1)
        kwargs.setdefault('offset_size', kwargs.get('patch_size', 16) - 1)
        params = ChromaRadianceParams(**kwargs)
        self.params = params
        self.patch_size = params.patch_size
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.in_dim = params.in_dim
        self.out_dim = params.out_dim
        self.hidden_dim = params.hidden_dim
        self.n_layers = params.n_layers
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in_patch = operations.Conv2d(
            params.in_channels,
            params.hidden_size,
            kernel_size=params.patch_size,
            stride=params.patch_size,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)
        self.distilled_guidance_layer = Approximator(
                    in_dim=self.in_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.out_dim,
                    n_layers=self.n_layers,
                    dtype=dtype, device=device, operations=operations
                )
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    dtype=dtype, device=device, operations=operations,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )
        self.nerf_image_embedder = NerfEmbedder(
            in_channels=params.in_channels,
            hidden_size_input=params.nerf_hidden_size,
            max_freqs=params.nerf_max_freqs,
            dtype=params.nerf_embedder_dtype or dtype,
            device=device,
            operations=operations,
        )
        self.nerf_blocks = nn.ModuleList([
            NerfGLUBlock(
                hidden_size_s=params.hidden_size,
                hidden_size_x=params.nerf_hidden_size,
                mlp_ratio=params.nerf_mlp_ratio,
                dtype=dtype,
                device=device,
                operations=operations,
            ) for _ in range(params.nerf_depth)
        ])
        if params.nerf_final_head_type == "linear":
            self.nerf_final_layer = NerfFinalLayer(
                params.nerf_hidden_size,
                out_channels=params.in_channels,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        elif params.nerf_final_head_type == "conv":
            self.nerf_final_layer_conv = NerfFinalLayerConv(
                params.nerf_hidden_size,
                out_channels=params.in_channels,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        else:
            errstr = f"Unsupported nerf_final_head_type {params.nerf_final_head_type}"
            raise ValueError(errstr)
        self.skip_mmdit = []
        self.skip_dit = []
        self.lite = False

    @property
    def _nerf_final_layer(self) -> nn.Module:
        if self.params.nerf_final_head_type == "linear":
            return self.nerf_final_layer
        if self.params.nerf_final_head_type == "conv":
            return self.nerf_final_layer_conv
        raise NotImplementedError

    def img_in(self, img: Tensor) -> Tensor:
        img = self.img_in_patch(img)
        return img.flatten(2).transpose(1, 2)

    def forward_nerf(
        self,
        img_orig: Tensor,
        img_out: Tensor,
        params: ChromaRadianceParams,
    ) -> Tensor:
        B, C, H, W = img_orig.shape
        num_patches = img_out.shape[1]
        patch_size = params.patch_size
        nerf_pixels = nn.functional.unfold(img_orig, kernel_size=patch_size, stride=patch_size)
        nerf_pixels = nerf_pixels.transpose(1, 2) # -> [B, NumPatches, C * P * P]

        # Reshape for per-patch processing
        nerf_hidden = img_out.reshape(B * num_patches, params.hidden_size)
        nerf_pixels = nerf_pixels.reshape(B * num_patches, C, patch_size**2).transpose(1, 2)

        if params.nerf_tile_size > 0 and num_patches > params.nerf_tile_size:
            # Enable tiling if nerf_tile_size isn't 0 and we actually have more patches than
            # the tile size.
            img_dct = self.forward_tiled_nerf(nerf_hidden, nerf_pixels, B, C, num_patches, patch_size, params)
        else:
            # Get DCT-encoded pixel embeddings [pixel-dct]
            img_dct = self.nerf_image_embedder(nerf_pixels)
            for block in self.nerf_blocks:
                img_dct = block(img_dct, nerf_hidden)
        img_dct = img_dct.transpose(1, 2)
        img_dct = img_dct.reshape(B, num_patches, -1)
        img_dct = img_dct.transpose(1, 2)
        img_dct = nn.functional.fold(
            img_dct,
            output_size=(H, W),
            kernel_size=patch_size,
            stride=patch_size,
        )
        return self._nerf_final_layer(img_dct)

    def forward_tiled_nerf(
        self,
        nerf_hidden: Tensor,
        nerf_pixels: Tensor,
        batch: int,
        channels: int,
        num_patches: int,
        patch_size: int,
        params: ChromaRadianceParams,
    ) -> Tensor:
        tile_size = params.nerf_tile_size
        output_tiles = []
        for i in range(0, num_patches, tile_size):
            end = min(i + tile_size, num_patches)

            # Slice the current tile from the input tensors
            nerf_hidden_tile = nerf_hidden[i * batch:end * batch]
            nerf_pixels_tile = nerf_pixels[i * batch:end * batch]

            # get DCT-encoded pixel embeddings [pixel-dct]
            img_dct_tile = self.nerf_image_embedder(nerf_pixels_tile)
            for block in self.nerf_blocks:
                img_dct_tile = block(img_dct_tile, nerf_hidden_tile)
            output_tiles.append(img_dct_tile)
        return torch.cat(output_tiles, dim=0)

    def radiance_get_override_params(self, overrides: dict) -> ChromaRadianceParams:
        params = self.params
        if not overrides:
            return params
        params_dict = {k: getattr(params, k) for k in params.__dataclass_fields__}
        nullable_keys = frozenset(("nerf_embedder_dtype",))
        bad_keys = tuple(k for k in overrides if k not in params_dict)
        if bad_keys:
            e = f"Unknown key(s) in transformer_options chroma_radiance_options: {', '.join(bad_keys)}"
            raise ValueError(e)
        bad_keys = tuple(
            k
            for k, v in overrides.items()
            if type(v) != type(getattr(params, k)) and (v is not None or k not in nullable_keys)
        )
        if bad_keys:
            e = f"Invalid value(s) in transformer_options chroma_radiance_options: {', '.join(bad_keys)}"
            raise ValueError(e)
        params_dict |= overrides
        return params.__class__(**params_dict)

    @staticmethod
    def _extract_crop_position_ids(full_img_ids, offset_y, offset_x, crop_height, crop_width, patch_size=16):
        batch_size = full_img_ids.shape[0]
        full_side_patches = int(full_img_ids.shape[1] ** 0.5)
        patch_offset_y = offset_y // patch_size
        patch_offset_x = offset_x // patch_size
        crop_patch_height = crop_height // patch_size
        crop_patch_width = crop_width // patch_size
        spatial_ids = full_img_ids.view(batch_size, full_side_patches, full_side_patches, 3)
        crop_ids = spatial_ids[:, patch_offset_y:patch_offset_y + crop_patch_height, patch_offset_x:patch_offset_x + crop_patch_width, :]
        return crop_ids.reshape(batch_size, -1, 3)

    def _prediction_pass(
        self,
        img: Tensor,
        img_ids: Tensor,
        context: Tensor,
        txt_ids: Tensor,
        timestep: Tensor,
        guidance: Optional[Tensor],
        control: Optional[dict],
        transformer_options: dict,
        params: ChromaRadianceParams,
        **kwargs: dict,
    ) -> Tensor:
        h_orig, w_orig = img.shape[-2], img.shape[-1]
        padded_img = comfy.ldm.common_dit.pad_to_patch_size(img, (self.patch_size, self.patch_size))
        img_out = self.forward_orig(
            padded_img,
            img_ids,
            context,
            txt_ids,
            timestep,
            guidance,
            control,
            transformer_options,
            attn_mask=kwargs.get("attention_mask", None),
        )
        denoised_img = self.forward_nerf(padded_img, img_out, params)
        return denoised_img[:, :, :h_orig, :w_orig]

    def _forward(
        self,
        x: Tensor,
        timestep: Tensor,
        context: Tensor,
        guidance: Optional[Tensor],
        control: Optional[dict]=None,
        transformer_options: dict={},
        **kwargs: dict,
    ) -> Tensor:
        radiance_opts = transformer_options.get("chroma_radiance_options", {})
        params = self.radiance_get_override_params(radiance_opts)

        bs, c, h, w = x.shape
        h_len = h // self.patch_size
        w_len = w // self.patch_size
        img_ids_full = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids_full[:, :, 1] += torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids_full[:, :, 2] += torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids_full = repeat(img_ids_full, "h w c -> b (h w) c", b=bs)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

        if not params.grid_mitigation_enabled:
            return self._prediction_pass(x, img_ids_full, context, txt_ids, timestep, guidance, control, transformer_options, params, **kwargs)
        else:
            pred_full = self._prediction_pass(x, img_ids_full, context, txt_ids, timestep, guidance, control, transformer_options, params, **kwargs)

            crop_size_h = h - self.patch_size
            crop_size_w = w - self.patch_size
            crop_size_h = (crop_size_h // self.patch_size) * self.patch_size
            crop_size_w = (crop_size_w // self.patch_size) * self.patch_size

            if params.num_offsets == 1:
                max_offset_y = min(params.offset_size, h - crop_size_h)
                max_offset_x = min(params.offset_size, w - crop_size_w)
                offset_y = torch.randint(0, max_offset_y + 1, (1,)).item() if max_offset_y >= 0 else 0
                offset_x = torch.randint(0, max_offset_x + 1, (1,)).item() if max_offset_x >= 0 else 0
                x_cropped = x[:, :, offset_y:offset_y+crop_size_h, offset_x:offset_x+crop_size_w]
                img_ids_cropped = self._extract_crop_position_ids(img_ids_full, offset_y, offset_x, crop_size_h, crop_size_w, self.patch_size)
                pred_crop = self._prediction_pass(x_cropped, img_ids_cropped, context, txt_ids, timestep, guidance, control, transformer_options, params, **kwargs)
                final_pred = pred_full.clone()
                final_pred[:, :, offset_y:offset_y+crop_size_h, offset_x:offset_x+crop_size_w] = pred_crop
                return final_pred
            else:
                accumulated_crops = torch.zeros_like(pred_full)
                crop_weights = torch.zeros_like(pred_full)

                for _ in range(params.num_offsets):
                    max_offset_y = min(params.offset_size, h - crop_size_h)
                    max_offset_x = min(params.offset_size, w - crop_size_w)
                    offset_y = torch.randint(0, max_offset_y + 1, (1,)).item() if max_offset_y >= 0 else 0
                    offset_x = torch.randint(0, max_offset_x + 1, (1,)).item() if max_offset_x >= 0 else 0

                    x_cropped = x[:, :, offset_y:offset_y+crop_size_h, offset_x:offset_x+crop_size_w]
                    img_ids_cropped = self._extract_crop_position_ids(img_ids_full, offset_y, offset_x, crop_size_h, crop_size_w, self.patch_size)
                    pred_crop = self._prediction_pass(x_cropped, img_ids_cropped, context, txt_ids, timestep, guidance, control, transformer_options, params, **kwargs)

                    accumulated_crops[:, :, offset_y:offset_y+crop_size_h, offset_x:offset_x+crop_size_w] += pred_crop
                    crop_weights[:, :, offset_y:offset_y+crop_size_h, offset_x:offset_x+crop_size_w] += 1.0

                crop_mask = crop_weights > 0
                blended_crops = torch.where(crop_mask, accumulated_crops / torch.clamp(crop_weights, min=1.0), pred_full)

                final_pred = torch.where(crop_mask, blended_crops, pred_full)
                return final_pred