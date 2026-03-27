import comfy.text_encoders.llama
from comfy import sd1_clip
from .spiece_tokenizer import SPieceTokenizer
import comfy.utils


class Gemma3_270MTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = tokenizer_data.get("spiece_model", None)
        special_tokens = {}
        super().__init__(tokenizer, pad_with_end=False, embedding_size=640,
                         embedding_key='gemma3_270m', tokenizer_class=SPieceTokenizer,
                         has_end_token=False, pad_to_max_length=True,
                         max_length=99999999, min_length=128,
                         tokenizer_args={"add_bos": True, "add_eos": False, "special_tokens": special_tokens},
                         tokenizer_data=tokenizer_data)

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}


class DeCoTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
                         name="gemma3_270m", tokenizer=Gemma3_270MTokenizer)


class Gemma3_270MModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None,
                 attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx,
                         textmodel_json_config={}, dtype=dtype,
                         special_tokens={"start": 2, "pad": 0},
                         layer_norm_hidden_state=False,
                         model_class=comfy.text_encoders.llama.Gemma3_270M,
                         enable_attention_masks=attention_mask,
                         return_attention_masks=False,
                         model_options=model_options)


class DeCoTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(device=device, dtype=dtype, name="gemma3_270m",
                         clip_model=Gemma3_270MModel, model_options=model_options)


def gemma3_270m_detect(state_dict, prefix=""):
    out = {}
    norm_keys = ["{}model.layers.0.self_attn.q_norm.weight".format(prefix),
                 "{}model.norm.weight".format(prefix)]
    for norm_key in norm_keys:
        if norm_key in state_dict:
            out["dtype_llama"] = state_dict[norm_key].dtype
            break

    quant = comfy.utils.detect_layer_quantization(state_dict, prefix)
    if quant is not None:
        out["llama_quantization_metadata"] = quant

    return out


def te(dtype_llama=None, llama_quantization_metadata=None):
    class DeCoTEModel_(DeCoTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return DeCoTEModel_
