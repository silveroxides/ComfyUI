"""Krea 2 (K2) text encoder: Qwen3-VL-4B, 12-layer tap.

K2 conditions on a stack of hidden states from 12 layers of Qwen3-VL-4B
(reference taps ``hidden_states[2,5,8,...,35]``), kept as a ``(B, 12, seq, 2560)`` tensor and
consumed by the DiT's internal ``txtfusion`` adapter. Comfy carries conditioning as a 3D tensor,
so the 12-layer stack is flattened to ``(B, seq, 12*2560)`` here and unpacked inside the model.
"""

import numbers

import torch

import comfy.text_encoders.qwen3vl
from comfy import sd1_clip

# tap k == hidden_states[k] (no offset).
KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]

# Identical system template to Qwen-Image; Krea2 strips the system+user-opening prefix.
KREA2_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def _krea2_template_end(tok_pairs):
    image_position = next((i for i, pair in enumerate(tok_pairs) if isinstance(pair[0], dict) and pair[0].get("type") == "image"), len(tok_pairs))
    template_end = -1
    for i in range(image_position):
        if i + 2 >= len(tok_pairs):
            break
        values = [tok_pairs[j][0] for j in range(i, i + 3)]
        if all(not torch.is_tensor(value) and isinstance(value, numbers.Integral) for value in values) and values == [151644, 872, 198]:
            template_end = i + 3
    if template_end == -1:
        raise ValueError("Could not locate the Krea 2 user prompt template.")
    return template_end


class Krea2Tokenizer(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_4b")
        self.llama_template = KREA2_TEMPLATE  # conditioning template; image text-gen uses qwen3vl's default image template.

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=True, **kwargs):
        # Krea2 conditions on the no-think template; thinking=True drops the empty <think> block qwen3vl adds.
        return super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, images=images, prevent_empty_text=prevent_empty_text, thinking=thinking, **kwargs)


class Krea2Qwen3VLClipModel(comfy.text_encoders.qwen3vl.Qwen3VLClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=KREA2_TAP_LAYERS, layer_idx=None, dtype=dtype,
                         attention_mask=attention_mask, model_options=model_options, model_type="qwen3vl_4b")


class Krea2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b", clip_model=Krea2Qwen3VLClipModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs, template_end=-1):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)  # out: (B, 12, seq, 2560)
        tok_pairs = token_weight_pairs["qwen3vl_4b"][0]

        if template_end == -1:
            template_end = _krea2_template_end(tok_pairs)

        out = out[:, :, template_end:]

        b, n, seq, h = out.shape
        # Flatten the 12-layer axis into the feature dim: (B, seq, 12*2560). Unpacked in the model.
        out = out.permute(0, 2, 1, 3).reshape(b, seq, n * h)

        if "attention_mask" in extra:
            extra["attention_mask"] = extra["attention_mask"][:, template_end:]
            if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
                extra.pop("attention_mask")

        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class Krea2TEModel_(Krea2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Krea2TEModel_
