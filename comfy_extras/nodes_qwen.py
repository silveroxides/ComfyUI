import node_helpers
import comfy.utils
import math
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.model_management
import torch
import nodes


def _spatial_fusion_mask(height, width, num_sources, method, block_size, dither_ratio, device):
    rows = torch.arange(height, device=device).unsqueeze(1)
    columns = torch.arange(width, device=device).unsqueeze(0)

    if method == "spatial-checkerboard":
        return ((rows + columns) % num_sources).flatten()
    if method == "spatial-block-interleave":
        return ((rows // block_size + columns // block_size) % num_sources).flatten()
    if method == "spatial-dither-random":
        generator = torch.Generator(device=device).manual_seed(42)
        random = torch.rand((height, width), generator=generator, device=device)
        if num_sources == 2:
            return torch.where(random < dither_ratio, 0, 1).flatten()
        return (random * num_sources).long().flatten()
    raise ValueError(f"Unsupported visual fusion method: {method}")


def _visual_token_span(tokens, cond_length, visual_tokens):
    if len(tokens) != 1:
        raise ValueError("Visual fusion requires a Qwen3-VL or Krea2 text encoder.")

    token_pairs = next(iter(tokens.values()))[0]
    image_positions = [i for i, pair in enumerate(token_pairs) if isinstance(pair[0], dict) and pair[0].get("type") == "image"]
    if len(image_positions) != 1:
        raise ValueError("Visual fusion requires exactly one visual token block per encoding pass.")

    image_position = image_positions[0]
    if any(not isinstance(pair[0], (int, float)) for pair in token_pairs[image_position + 1:]):
        raise ValueError("Visual fusion does not support embeddings after the image token block.")

    end = cond_length - (len(token_pairs) - image_position - 1)
    start = end - visual_tokens
    if start < 0 or end > cond_length:
        raise ValueError("Could not locate the visual token block in the encoded conditioning.")
    return start, end


def _fuse_conditionings(conditionings, tokens, visual_height, visual_width, method, block_size, dither_ratio):
    schedule_count = len(conditionings[0])
    if any(len(source) != schedule_count for source in conditionings):
        raise ValueError("All visual fusion sources must use the same CLIP schedule.")

    visual_tokens = visual_height * visual_width
    fused = []
    for schedule in range(schedule_count):
        source_conds = [source[schedule][0] for source in conditionings]
        spans = [_visual_token_span(source_tokens, cond.shape[1], visual_tokens) for source_tokens, cond in zip(tokens, source_conds)]
        if any(span != spans[0] for span in spans[1:]):
            raise ValueError("Visual fusion sources produced different token layouts.")

        start, end = spans[0]
        visuals = torch.stack([cond[:, start:end] for cond in source_conds], dim=2)
        mask = _spatial_fusion_mask(visual_height, visual_width, len(source_conds), method, block_size, dither_ratio, visuals.device)
        blended_visual = torch.take_along_dim(visuals, mask[None, :, None, None], dim=2).squeeze(2)

        blended = source_conds[0].clone()
        blended[:, start:end] = blended_visual
        fused.append([blended, conditionings[0][schedule][1].copy()])
    return fused


def _flatten_images(images):
    sources = []
    for name in sorted(images, key=lambda value: int(value.rsplit("_", 1)[-1])):
        image = images[name]
        if image is None:
            continue
        if image.ndim == 3:
            image = image.unsqueeze(0)
        sources.extend(image[i:i + 1].clone() for i in range(image.shape[0]))
    return sources


class TextEncodeQwenImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEdit",
            category="model/conditioning/qwen image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image=None) -> io.NodeOutput:
        ref_latent = None
        if image is None:
            images = []
        else:
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
        return io.NodeOutput(conditioning)


class TextEncodeQwenImageEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEditPlus",
            category="model/conditioning/qwen image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning)


class TextEncodeQwenImageEditFusion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        images = io.Autogrow.TemplateNames(
            io.Image.Input("image"),
            names=[f"image_{i}" for i in range(1, 17)],
            min=2,
        )
        return io.Schema(
            node_id="TextEncodeQwenImageEditFusion",
            display_name="Text Encode Qwen Image Edit (Visual Fusion)",
            category="model/conditioning/qwen image",
            description="Encodes images separately and spatially interleaves their Qwen3-VL visual conditioning tokens.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Autogrow.Input("images", template=images),
                io.Combo.Input(
                    "fusion_method",
                    options=["spatial-checkerboard", "spatial-block-interleave", "spatial-dither-random"],
                    default="spatial-checkerboard",
                ),
                io.Int.Input("block_size", default=2, min=1, max=8, step=1, advanced=True),
                io.Float.Input(
                    "dither_ratio",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="For two sources, the probability of selecting the first source. Three or more sources are selected uniformly.",
                ),
                io.Vae.Input("vae", optional=True),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, prompt, images: io.Autogrow.Type, fusion_method, block_size=2, dither_ratio=0.5, vae=None) -> io.NodeOutput:
        sources = _flatten_images(images)
        if len(sources) < 2:
            raise ValueError("Visual fusion requires at least two images.")

        first = sources[0].movedim(-1, 1)
        total = 384 * 384
        scale_by = math.sqrt(total / (first.shape[3] * first.shape[2]))
        width = max(32, round(first.shape[3] * scale_by))
        height = max(32, round(first.shape[2] * scale_by))

        processed = []
        for source in sources:
            samples = source[:, :, :, :3].movedim(-1, 1)
            resized = comfy.utils.common_upscale(samples, width, height, "area", "center")
            processed.append(resized.movedim(1, -1))

        factor = 32
        visual_height = max(factor, round(height / factor) * factor) // factor
        visual_width = max(factor, round(width / factor) * factor) // factor

        full_prompt = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokens = [clip.tokenize(full_prompt, images=[image]) for image in processed]
        token_key = next(iter(tokens[0]), None)
        if token_key not in ("qwen3vl_4b", "qwen3vl_8b") or any(next(iter(source_tokens), None) != token_key for source_tokens in tokens):
            raise ValueError("Visual fusion requires a Qwen3-VL or Krea2 text encoder.")

        conditionings = [clip.encode_from_tokens_scheduled(source_tokens) for source_tokens in tokens]
        conditioning = _fuse_conditionings(conditionings, tokens, visual_height, visual_width, fusion_method, block_size, dither_ratio)

        if vae is not None:
            ref_latents = []
            for source in sources:
                samples = source[:, :, :, :3].movedim(-1, 1)
                scale_by = math.sqrt((1024 * 1024) / (samples.shape[3] * samples.shape[2]))
                latent_width = max(8, round(samples.shape[3] * scale_by / 8.0) * 8)
                latent_height = max(8, round(samples.shape[2] * scale_by / 8.0) * 8)
                resized = comfy.utils.common_upscale(samples, latent_width, latent_height, "area", "disabled")
                ref_latents.append(vae.encode(resized.movedim(1, -1)))
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)

        return io.NodeOutput(conditioning)


class EmptyQwenImageLayeredLatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyQwenImageLayeredLatentImage",
            display_name="Empty Qwen Image Layered Latent",
            category="model/latent/qwen",
            inputs=[
                io.Int.Input("width", default=640, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=640, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("layers", default=3, min=0, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, layers, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, layers + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent})


class QwenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeQwenImageEdit,
            TextEncodeQwenImageEditPlus,
            TextEncodeQwenImageEditFusion,
            EmptyQwenImageLayeredLatentImage,
        ]


async def comfy_entrypoint() -> QwenExtension:
    return QwenExtension()
