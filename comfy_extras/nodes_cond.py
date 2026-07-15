import re

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


_IMAGE_INPUT_PATTERN = re.compile(r"\bimage_input_(\d+)\b", re.IGNORECASE)


def _image_token_count(tokens):
    return sum(isinstance(pair[0], dict) and pair[0].get("type") == "image" for rows in tokens.values() for row in rows for pair in row)


class CLIPTextEncodeImagePlaceholders(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        images = io.Autogrow.TemplateNames(
            io.Image.Input("image"),
            names=[f"image_{i}" for i in range(1, 17)],
            min=1,
        )
        return io.Schema(
            node_id="CLIPTextEncodeImagePlaceholders",
            display_name="CLIP Text Encode (Image Placeholders)",
            category="model/conditioning",
            description="Encodes images at image_input_N locations in the text. Image resolution affects conditioning quality and memory use.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Autogrow.Input("images", template=images),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, text, images: io.Autogrow.Type) -> io.NodeOutput:
        placeholders = [match.group(0).lower() for match in _IMAGE_INPUT_PATTERN.finditer(text)]
        if len(placeholders) == 0:
            raise ValueError("Image placeholder encoding requires at least one image_input_N placeholder.")

        inline_images = {}
        for name, image in (images or {}).items():
            if image is not None:
                number = int(name.rsplit("_", 1)[-1])
                inline_images[f"image_input_{number}"] = image[:, :, :, :3]

        for placeholder in placeholders:
            if placeholder not in inline_images:
                raise ValueError(f"No image is connected for {placeholder}.")

        expected_images = sum(inline_images[placeholder].shape[0] for placeholder in placeholders)
        tokens = clip.tokenize(text, inline_images=inline_images)
        if _image_token_count(tokens) != expected_images:
            raise ValueError("The selected CLIP tokenizer does not support inline images.")

        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))


class CLIPTextEncodeControlnet(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPTextEncodeControlnet",
            display_name="CLIP Text Encode (Controlnet)",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
            ],
            outputs=[io.Conditioning.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, conditioning, text) -> io.NodeOutput:
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['cross_attn_controlnet'] = cond
            n[1]['pooled_output_controlnet'] = pooled
            c.append(n)
        return io.NodeOutput(c)

class T5TokenizerOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T5TokenizerOptions",
            display_name="T5 Tokenizer Options",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("min_padding", default=0, min=0, max=10000, step=1),
                io.Int.Input("min_length", default=0, min=0, max=10000, step=1),
            ],
            outputs=[io.Clip.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, min_padding, min_length) -> io.NodeOutput:
        clip = clip.clone()
        for t5_type in ["t5xxl", "pile_t5xl", "t5base", "mt5xl", "umt5xxl"]:
            clip.set_tokenizer_option("{}_min_padding".format(t5_type), min_padding)
            clip.set_tokenizer_option("{}_min_length".format(t5_type), min_length)

        return io.NodeOutput(clip)


class CondExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeImagePlaceholders,
            CLIPTextEncodeControlnet,
            T5TokenizerOptions,
        ]


async def comfy_entrypoint() -> CondExtension:
    return CondExtension()
