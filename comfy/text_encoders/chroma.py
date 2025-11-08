from comfy import sd1_clip
import comfy.text_encoders.t5
import comfy.text_encoders.sd3_clip
import comfy.model_management
from comfy.text_encoders.sd3_clip import T5XXLModel
from comfy.text_encoders.qwen_image import QwenImageTokenizer, QwenImageTEModel
from transformers import T5TokenizerFast, T5EncoderModel, T5Config, T5PreTrainedModel
import torch
import os
import logging

class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, min_length=1):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=min_length, tokenizer_data=tokenizer_data)


class ChromaTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.qwen25_7b = QwenImageTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids)
        out["qwen25_7b"] = self.qwen25_7b.tokenize_with_weights(text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class ChromaTEModel(torch.nn.Module):
    def __init__(self, clip_l=True, t5=True, qwen25_7b=True, dtype_t5=None, dtype_llama=None, t5_attention_mask=False, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.dtypes = set()
        if clip_l:
            self.clip_l = sd1_clip.SDClipModel(layer="hidden", layer_idx=-2, device=device, dtype=dtype, layer_norm_hidden_state=False, return_projected_pooled=False, model_options=model_options)
            self.dtypes.add(dtype)
        else:
            self.clip_l = None

        if t5:
            dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
            self.t5_attention_mask = t5_attention_mask
            self.t5xxl = T5XXLModel(device=device, dtype=dtype_t5, model_options=model_options, attention_mask=self.t5_attention_mask)
            self.dtypes.add(dtype_t5)
        else:
            self.t5xxl = None

        if qwen25_7b:
            dtype_llama = comfy.model_management.pick_weight_dtype(dtype_llama, dtype, device)
            self.qwen25_7b = QwenImageTEModel(device=device, dtype=dtype_llama, model_options=model_options)
            self.dtypes.add(dtype_llama)
        else:
            self.qwen25_7b = None

        logging.debug("Created Chroma text encoder with: clip_l {}, t5xxl {}:{}".format(clip_l, t5, dtype_t5))

    def set_clip_options(self, options):
        if self.clip_l is not None:
            self.clip_l.set_clip_options(options)
        if self.t5xxl is not None:
            self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        if self.clip_l is not None:
            self.clip_l.reset_clip_options()
        if self.t5xxl is not None:
            self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]
        token_weight_pairs_q = token_weight_pairs["qwen25_7b"]
        l_out = None
        q_out = None
        pooled = None
        out = None
        extra = {}

        if len(token_weight_pairs_l) > 0:
            if self.clip_l is not None:
                l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
                l_out = torch.nn.functional.pad(l_out, (0, 4096 - l_out.shape[-1]))
                out = l_out
            else:
                l_out = None
                l_pooled = torch.zeros((1, 768), device=comfy.model_management.intermediate_device())

        if len(token_weight_pairs_q) > 0:
            if self.qwen25_7b is not None:
                q_out, q_pooled, extra = self.qwen25_7b.encode_token_weights(token_weight_pairs_q)
                q_out = torch.nn.functional.pad(q_out, (0, 4096 - q_out.shape[-1]))
                out = q_out
            else:
                q_out = None

        if self.t5xxl is not None:
            t5_output = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
            t5_out, t5_pooled = t5_output[:2]
            if self.t5_attention_mask:
                extra["attention_mask"] = t5_output[2]["attention_mask"]

            if l_out is not None:
                out = torch.cat([l_out, t5_out], dim=-2)
            elif q_out is not None:
                out = torch.cat([q_out, t5_out], dim=-2)
            else:
                out = t5_out


        if out is None:
            out = torch.zeros((1, 1, 4096), device=comfy.model_management.intermediate_device())

        if pooled is None:
            pooled = torch.zeros((1, 768), device=comfy.model_management.intermediate_device())

        return out, pooled, extra

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        elif "model.layers.1.post_attention_layernorm.weight" in sd:
            return self.qwen25_7b.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)

def chroma_te(clip_l=True, t5=True, qwen25_7b=True, dtype_t5=None, dtype_llama=None, t5xxl_scaled_fp8=None, llama_scaled_fp8=None, t5_attention_mask=False):
    class ChromaTEModel_(ChromaTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if t5xxl_scaled_fp8 is not None and "t5xxl_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["t5xxl_scaled_fp8"] = t5xxl_scaled_fp8
            if llama_scaled_fp8 is not None and "scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["scaled_fp8"] = llama_scaled_fp8
            super().__init__(clip_l=clip_l, t5=t5, qwen25_7b=qwen25_7b, dtype_t5=dtype_t5, dtype_llama=dtype_llama, t5_attention_mask=t5_attention_mask, device=device, dtype=dtype, model_options=model_options)
    return ChromaTEModel_
