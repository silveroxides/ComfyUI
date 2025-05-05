from comfy import sd1_clip
import comfy.text_encoders.t5
import comfy.text_encoders.sd3_clip
import comfy.model_management
from transformers import T5TokenizerFast, T5EncoderModel, T5Config, T5PreTrainedModel
import torch
import os

class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, min_length=1):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=min_length, tokenizer_data=tokenizer_data)


class ChromaDuoTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class ChromaDuoTEModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, t5_attention_mask=False, device="cpu", dtype=None, model_options={}):
        super().__init__()
        dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
        self.clip_l = sd1_clip.SDClipModel(layer="hidden", layer_idx=-1, device=device, dtype=dtype, layer_norm_hidden_state=False, return_projected_pooled=False, model_options=model_options)
        self.t5_attention_mask = t5_attention_mask
        self.t5xxl = comfy.text_encoders.sd3_clip.T5XXLModel(device=device, dtype=dtype_t5, model_options=model_options, attention_mask=self.t5_attention_mask)
        self.dtypes = set([dtype, dtype_t5])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]

        t5_out, t5_pooled = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        print(f"t5_out={t5_out}, t5_pooled={t5_pooled}, l_out={l_out}, l_pooled={l_pooled}")
        cut_to = min(l_out.shape[1], t5_out.shape[1])
        lt5_out = torch.cat([l_out[:,:cut_to], t5_out[:,:cut_to]], dim=-1)
        ll_out = torch.nn.functional.pad(lt5_out, (0, t5_out.shape[-1] - lt5_out.shape[-1]))
        t5_out = torch.cat([ll_out, t5_out], dim=-2)
        t5_pooled = torch.zeros((1, 4096), device=comfy.model_management.intermediate_device())
        return t5_out, t5_out

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)

def chromaduo_te(dtype_t5=None, t5xxl_scaled_fp8=None, t5_attention_mask=False):
    class ChromaDuoTEModel_(ChromaDuoTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if t5xxl_scaled_fp8 is not None and "t5xxl_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["t5xxl_scaled_fp8"] = t5xxl_scaled_fp8
            super().__init__(dtype_t5=dtype_t5, t5_attention_mask=t5_attention_mask, device=device, dtype=dtype, model_options=model_options)
    return ChromaDuoTEModel_
