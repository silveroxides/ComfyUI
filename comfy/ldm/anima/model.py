from comfy.ldm.cosmos.predict2 import MiniTrainDIT
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import comfy.ldm.common_dit


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim, device=None, dtype=None, operations=None):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = operations.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)

        self.k_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)

        self.v_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)

        self.o_proj = operations.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def init_weights(self):
        torch.nn.init.zeros_(self.o_proj.weight)


class TransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=False, layer_norm=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.norm_self_attn = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
            self.self_attn = Attention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim//num_heads,
                device=device,
                dtype=dtype,
                operations=operations,
            )

        self.norm_cross_attn = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = Attention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim//num_heads,
            device=device,
            dtype=dtype,
            operations=operations,
        )

        self.norm_mlp = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            operations.Linear(model_dim, int(model_dim * mlp_ratio), device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(int(model_dim * mlp_ratio), model_dim, device=device, dtype=dtype)
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(normed, mask=target_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings)
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(normed, mask=source_attention_mask, context=context, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)
        self.cross_attn.init_weights()


class LLMAdapter(nn.Module):
    def __init__(
            self,
            source_dim=1024,
            target_dim=1024,
            model_dim=1024,
            num_layers=6,
            num_heads=16,
            use_self_attn=True,
            layer_norm=False,
            device=None,
            dtype=None,
            operations=None,
        ):
        super().__init__()

        self.embed = operations.Embedding(32128, target_dim, device=device, dtype=dtype)
        if model_dim != target_dim:
            self.in_proj = operations.Linear(target_dim, model_dim, device=device, dtype=dtype)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim//num_heads)
        self.blocks = nn.ModuleList([
            TransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn, layer_norm=layer_norm, device=device, dtype=dtype, operations=operations) for _ in range(num_layers)
        ])
        self.out_proj = operations.Linear(model_dim, target_dim, device=device, dtype=dtype)
        self.norm = operations.RMSNorm(target_dim, eps=1e-6, device=device, dtype=dtype)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids, out_dtype=context.dtype))
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(x, context, target_attention_mask=target_attention_mask, source_attention_mask=source_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))


class Anima(MiniTrainDIT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_adapter = LLMAdapter(device=kwargs.get("device"), dtype=kwargs.get("dtype"), operations=kwargs.get("operations"))

    def preprocess_text_embeds(self, text_embeds, text_ids, t5xxl_weights=None):
        if text_ids is not None:
            out = self.llm_adapter(text_embeds, text_ids)
            if t5xxl_weights is not None:
                out = out * t5xxl_weights

            if out.shape[1] < 512:
                out = torch.nn.functional.pad(out, (0, 0, 0, 512 - out.shape[1]))
            return out
        else:
            return text_embeds

    def _rope_for_region(self, t_offset, T, H, W, fps, device):
        """Compute VideoRopePosition3DEmb-compatible rope for a (T, H, W) region
        with temporal positions starting at t_offset. Mirrors generate_embeddings
        but uses an explicit temporal range instead of seq[:T]."""
        pe = self.pos_embedder
        h_theta = 10000.0 * pe.h_ntk_factor
        w_theta = 10000.0 * pe.w_ntk_factor
        t_theta = 10000.0 * pe.t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** pe.dim_spatial_range.to(device=device))
        w_spatial_freqs = 1.0 / (w_theta ** pe.dim_spatial_range.to(device=device))
        temporal_freqs = 1.0 / (t_theta ** pe.dim_temporal_range.to(device=device))

        seq_hw = torch.arange(max(H, W), dtype=torch.float, device=device)
        half_emb_h = torch.outer(seq_hw[:H], h_spatial_freqs)
        half_emb_w = torch.outer(seq_hw[:W], w_spatial_freqs)

        if fps is None or not pe.enable_fps_modulation:
            seq_t = torch.arange(t_offset, t_offset + T, dtype=torch.float, device=device)
        else:
            fps_val = fps if isinstance(fps, (int, float)) else fps.item()
            seq_t = torch.arange(t_offset, t_offset + T, dtype=torch.float, device=device) / fps_val * pe.base_fps

        half_emb_t = torch.outer(seq_t, temporal_freqs)

        def _to_rotary(e):
            return torch.stack([torch.cos(e), -torch.sin(e), torch.sin(e), torch.cos(e)], dim=-1)

        half_emb_h = _to_rotary(half_emb_h)
        half_emb_w = _to_rotary(half_emb_w)
        half_emb_t = _to_rotary(half_emb_t)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
                repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
                repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )
        return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()

    def process_latent(self, ref, t_offset, target_h, target_w, fps, device):
        """Patchify a reference latent and compute its RoPE with temporal offset.

        Args:
            ref: (B, C, T, H, W) reference latent, already on correct device/dtype.
            t_offset: temporal index assigned to this ref's first frame in rope space.
            target_h, target_w: spatial dims of gen latent (pad ref if smaller).
            fps: fps tensor or None.
            device: torch device.

        Returns:
            embedded: (B, T_p, H_p, W_p, D) patch-embedded tokens.
            rope: (T_p*H_p*W_p, d, 2, 2) rope for these tokens.
            num_tokens: T_p * H_p * W_p.
        """
        B, C, T, H, W = ref.shape

        # pad spatial to match gen resolution
        if H < target_h or W < target_w:
            pad_h = target_h - H
            pad_w = target_w - W
            ref = F.pad(ref, (0, pad_w, 0, pad_h))
            H, W = target_h, target_w

        # pad temporal/spatial to patch multiples
        ref = comfy.ldm.common_dit.pad_to_patch_size(ref, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
        _, _, T_pad, H_pad, W_pad = ref.shape

        # add padding mask channel (zeros = unmasked) if model expects it
        if self.concat_padding_mask:
            pad_mask = torch.zeros(B, 1, H_pad, W_pad, dtype=ref.dtype, device=device)
            ref = torch.cat([ref, pad_mask.unsqueeze(1).repeat(1, 1, T_pad, 1, 1)], dim=1)

        # embed patches: (B, T_p, H_p, W_p, D)
        embedded = self.x_embedder(ref)
        _, T_p, H_p, W_p, _ = embedded.shape

        rope = self._rope_for_region(t_offset, T_p, H_p, W_p, fps, device)
        num_tokens = T_p * H_p * W_p
        return embedded, rope, num_tokens

    def _forward(self, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        ref_latents = kwargs.pop('ref_latents', None)
        ref_method = kwargs.pop('ref_latents_method', 'index')
        orig_shape = list(x.shape)

        if ref_latents is None:
            # no refs — delegate entirely to parent
            return super()._forward(x, timesteps, context, fps=fps, padding_mask=padding_mask, **kwargs)

        # --- embed gen tokens ---
        x_padded = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
        x_gen, rope_gen, extra_pos = self.prepare_embedded_sequence(x_padded, fps=fps, padding_mask=padding_mask)
        # x_gen: (B, T_gp, H_gp, W_gp, D)
        B, T_gp, H_gp, W_gp, D = x_gen.shape
        target_h = x_padded.shape[3]
        target_w = x_padded.shape[4]

        # --- embed ref tokens ---
        index = 0
        ref_embedded_list = []
        ref_rope_list = []
        ref_num_tokens = []

        for ref in ref_latents:
            if ref.ndim == 4:
                ref = ref.unsqueeze(2)
            ref = ref.to(dtype=x.dtype, device=x.device)

            if ref_method in ('index', 'index_timestep_zero'):
                index += 1
                t_off = index * T_gp  # each ref at next T block
            elif ref_method == 'negative_index':
                index -= 1
                # place refs at high T positions far from gen (T=0)
                max_t = self.max_frames // self.patch_temporal
                t_off = max_t + index  # index is negative so t_off < max_t
            else:
                index += 1
                t_off = index * T_gp

            emb, rope, ntok = self.process_latent(ref, t_off, target_h, target_w, fps, x.device)
            ref_embedded_list.append(emb)
            ref_rope_list.append(rope)
            ref_num_tokens.append(ntok)

        # --- cat tokens along T dim (keeps 5D structure for blocks) ---
        # All refs padded to same spatial as gen, so cat on T is valid
        all_embedded = torch.cat([x_gen] + ref_embedded_list, dim=1)  # (B, T_total_p, H_gp, W_gp, D)
        all_rope = torch.cat([rope_gen] + ref_rope_list, dim=0)        # (L_total, d, 2, 2)

        # --- timestep embedding ---
        timesteps_B_T = timesteps
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](self.t_embedder[0](timesteps_B_T).to(all_embedded.dtype))
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        T_total = all_embedded.shape[1]
        T_ref_total = T_total - T_gp

        if ref_method == 'index_timestep_zero' and T_ref_total > 0:
            # ref frames see zero-noise timestep: compute embedding at sigma=0
            zero_ts = torch.zeros_like(timesteps_B_T)
            t_emb_zero, adaln_zero = self.t_embedder[1](self.t_embedder[0](zero_ts).to(all_embedded.dtype))
            t_emb_zero = self.t_embedding_norm(t_emb_zero)
            # expand gen embedding to T_gp, zero embedding to T_ref_total, then cat
            t_gen = t_embedding_B_T_D.expand(-1, T_gp, -1)
            t_ref = t_emb_zero.expand(-1, T_ref_total, -1)
            t_embedding_B_T_D = torch.cat([t_gen, t_ref], dim=1)
            if adaln_lora_B_T_3D is not None:
                adaln_ref = adaln_zero.expand(-1, T_ref_total, -1) if adaln_zero is not None else torch.zeros(B, T_ref_total, adaln_lora_B_T_3D.shape[-1], dtype=adaln_lora_B_T_3D.dtype, device=adaln_lora_B_T_3D.device)
                adaln_lora_B_T_3D = torch.cat([adaln_lora_B_T_3D.expand(-1, T_gp, -1), adaln_ref], dim=1)
        else:
            # all T positions share the same timestep embedding
            if t_embedding_B_T_D.shape[1] == 1:
                t_embedding_B_T_D = t_embedding_B_T_D.expand(-1, T_total, -1)
            if adaln_lora_B_T_3D is not None and adaln_lora_B_T_3D.shape[1] == 1:
                adaln_lora_B_T_3D = adaln_lora_B_T_3D.expand(-1, T_total, -1)

        # --- transformer_options ---
        transformer_options = dict(kwargs.get('transformer_options', {}))
        transformer_options['reference_image_num_tokens'] = ref_num_tokens
        kwargs = dict(kwargs)
        kwargs['transformer_options'] = transformer_options

        # pad extra_pos to cover ref T slices with zeros if present
        if extra_pos is not None:
            T_extra = all_embedded.shape[1] - extra_pos.shape[1]
            if T_extra > 0:
                extra_pos = torch.cat(
                    [extra_pos, torch.zeros(B, T_extra, H_gp, W_gp, D, dtype=extra_pos.dtype, device=extra_pos.device)],
                    dim=1,
                )

        block_kwargs = {
            "rope_emb_L_1_1_D": all_rope.unsqueeze(1).unsqueeze(0),
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos,
            "transformer_options": transformer_options,
        }

        x_B_T_H_W_D = all_embedded
        if x_B_T_H_W_D.dtype == torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        for block in self.blocks:
            x_B_T_H_W_D = block(x_B_T_H_W_D, t_embedding_B_T_D, context, **block_kwargs)

        # final layer on all tokens then crop to gen tokens only
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D.to(context.dtype), t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        # crop to gen T frames and unpatchify
        x_B_T_H_W_O = x_B_T_H_W_O[:, :T_gp]
        x_out = self.unpatchify(x_B_T_H_W_O)
        return x_out[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]

    def forward(self, x, timesteps, context, **kwargs):
        t5xxl_ids = kwargs.pop("t5xxl_ids", None)
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights=kwargs.pop("t5xxl_weights", None))
        return super().forward(x, timesteps, context, **kwargs)
