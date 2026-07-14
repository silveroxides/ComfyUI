import node_helpers
import comfy.utils
import comfy.text_encoders.krea2
import folder_paths
import json
import math
import os
import time
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.model_management
import torch
import nodes


Krea2Experiment = io.Custom("KREA2_EXPERIMENT")


KREA2_EXPERIMENT_CONFIG = """[
  {
    "case_name": "1024-full",
    "prompt": "Apply the requested edit while preserving everything else.",
    "negative_prompt": "",
    "visual_resolution": 1024,
    "visual_token_ratio": 1.0,
    "selection_method": "uniform-grid",
    "seed": 1,
    "steps": 20,
    "cfg": 1.0,
    "sampler_name": "euler",
    "scheduler": "normal",
    "denoise": 1.0,
    "width": 1024,
    "height": 1024
  }
]"""


def _spatial_fusion_mask(height, width, num_sources, method, block_size, dither_ratio, device, seed=0):
    rows = torch.arange(height, device=device).unsqueeze(1)
    columns = torch.arange(width, device=device).unsqueeze(0)

    if method == "spatial-checkerboard":
        return ((rows + columns) % num_sources).flatten()
    if method == "spatial-block-interleave":
        return ((rows // block_size + columns // block_size) % num_sources).flatten()
    if method == "spatial-dither-random":
        generator = torch.Generator(device=device).manual_seed(seed)
        random = torch.rand((height, width), generator=generator, device=device)
        other_sources = 1 + ((rows + columns) % (num_sources - 1))
        return torch.where(random < dither_ratio, 0, other_sources).flatten()
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


def _fuse_conditionings(conditionings, tokens, visual_height, visual_width, method, block_size, dither_ratio, seed=0):
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
        mask = _spatial_fusion_mask(visual_height, visual_width, len(source_conds), method, block_size, dither_ratio, visuals.device, seed)
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


def _visual_token_indices(grid_height, grid_width, ratio, method):
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("Visual token ratio must be between 0.0 and 1.0.")
    total = grid_height * grid_width
    selected = round(total * ratio)
    if selected == 0:
        return []
    if method == "uniform-grid":
        selected_rows = round(math.sqrt(selected * grid_height / grid_width))
        selected_rows = min(grid_height, selected, max(math.ceil(selected / grid_width), selected_rows, 1))
        rows = [math.floor((i + 0.5) * grid_height / selected_rows) for i in range(selected_rows)]
        base_columns, extra_rows = divmod(selected, selected_rows)
        extra = set(math.floor((i + 0.5) * selected_rows / extra_rows) for i in range(extra_rows)) if extra_rows else set()
        indices = []
        for i, row in enumerate(rows):
            column_count = base_columns + (i in extra)
            columns = [math.floor((j + 0.5) * grid_width / column_count) for j in range(column_count)]
            indices.extend(row * grid_width + column for column in columns)
        return sorted(indices)
    if method == "legacy-tail":
        return list(range(total - selected, total))
    raise ValueError(f"Unsupported visual token selection method: {method}")


def _select_visual_tokens(conditioning, tokens, visual_tokens, selected_indices):
    selected_indices = list(selected_indices)
    output = []
    for cond, metadata in conditioning:
        start, end = _visual_token_span(tokens, cond.shape[1], visual_tokens)
        keep = list(range(start)) + [start + index for index in selected_indices] + list(range(end, cond.shape[1]))
        selected = cond[:, keep]
        selected_metadata = metadata.copy()
        attention_mask = selected_metadata.get("attention_mask")
        if attention_mask is not None:
            selected_metadata["attention_mask"] = attention_mask[:, keep]
        output.append([selected, selected_metadata])
    return output


def _visual_token_diagnostic(image, grid_height, grid_width, selected_indices):
    diagnostic = image.clone()
    selected = set(selected_indices)
    height, width = image.shape[1:3]
    for index in range(grid_height * grid_width):
        row, column = divmod(index, grid_width)
        top = round(row * height / grid_height)
        bottom = round((row + 1) * height / grid_height)
        left = round(column * width / grid_width)
        right = round((column + 1) * width / grid_width)
        if index not in selected:
            diagnostic[:, top:bottom, left:right] *= 0.2
        else:
            diagnostic[:, top:min(top + 2, bottom), left:right] = 1.0
            diagnostic[:, max(top, bottom - 2):bottom, left:right] = 1.0
            diagnostic[:, top:bottom, left:min(left + 2, right)] = 1.0
            diagnostic[:, top:bottom, max(left, right - 2):right] = 1.0
    return diagnostic


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


class TextEncodeKrea2VisualTokenControl(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeKrea2VisualTokenControl",
            display_name="Text Encode Krea 2 Visual Token Control",
            category="model/conditioning/qwen image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Image.Input("image"),
                io.Combo.Input("visual_resolution", options=[384, 512, 768, 1024], default=1024),
                io.Float.Input("visual_token_ratio", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Combo.Input("selection_method", options=["uniform-grid", "legacy-tail"], default="uniform-grid"),
            ],
            outputs=[io.Conditioning.Output(), io.Image.Output(), Krea2Experiment.Output()],
        )

    @classmethod
    def execute(cls, clip, prompt, image, visual_resolution=1024, visual_token_ratio=1.0, selection_method="uniform-grid") -> io.NodeOutput:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Krea 2 visual token control requires exactly one image.")
        if not isinstance(clip.cond_stage_model, comfy.text_encoders.krea2.Krea2TEModel):
            raise ValueError("Krea 2 visual token control requires a Krea 2 text encoder.")

        samples = image[:, :, :, :3].movedim(-1, 1)
        scale_by = math.sqrt((visual_resolution * visual_resolution) / (samples.shape[2] * samples.shape[3]))
        width = max(32, round(samples.shape[3] * scale_by))
        height = max(32, round(samples.shape[2] * scale_by))
        resized = comfy.utils.common_upscale(samples, width, height, "area", "disabled").movedim(1, -1)

        grid_height = max(1, round(height / 32))
        grid_width = max(1, round(width / 32))
        visual_tokens = grid_height * grid_width
        selected_indices = _visual_token_indices(grid_height, grid_width, visual_token_ratio, selection_method)

        full_prompt = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokens = clip.tokenize(full_prompt, images=[resized])
        encode_started = time.perf_counter()
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        encode_seconds = time.perf_counter() - encode_started
        conditioning = _select_visual_tokens(conditioning, tokens, visual_tokens, selected_indices)
        diagnostic = _visual_token_diagnostic(resized, grid_height, grid_width, selected_indices)
        experiment = {
            "prompt": prompt,
            "visual_resolution": visual_resolution,
            "visual_token_ratio": visual_token_ratio,
            "selection_method": selection_method,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "total_visual_tokens": visual_tokens,
            "selected_visual_tokens": len(selected_indices),
            "encode_seconds": encode_seconds,
        }
        return io.NodeOutput(conditioning, diagnostic, experiment)


class Krea2ExperimentConfiguration(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        outputs = [
            io.AnyType.Output(display_name=name) for name in (
                "prompt", "negative_prompt", "visual_resolution", "visual_token_ratio", "selection_method",
                "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "width", "height", "case_name",
            )
        ]
        outputs.append(Krea2Experiment.Output(display_name="configuration"))
        return io.Schema(
            node_id="Krea2ExperimentConfiguration",
            display_name="Krea 2 Experiment Configuration",
            category="model/conditioning/qwen image",
            inputs=[
                io.String.Input("configurations", default=KREA2_EXPERIMENT_CONFIG, multiline=True),
                io.Int.Input("index", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=outputs,
        )

    @classmethod
    def execute(cls, configurations, index=0) -> io.NodeOutput:
        values = json.loads(configurations)
        if not isinstance(values, list) or not values:
            raise ValueError("Krea 2 experiment configurations must be a non-empty JSON list.")
        if index >= len(values):
            raise ValueError(f"Krea 2 experiment configuration index {index} is out of range for {len(values)} configurations.")
        configuration = values[index]
        if not isinstance(configuration, dict):
            raise ValueError("Each Krea 2 experiment configuration must be a JSON object.")
        required = (
            "prompt", "negative_prompt", "visual_resolution", "visual_token_ratio", "selection_method",
            "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "width", "height", "case_name",
        )
        missing = [name for name in required if name not in configuration]
        if missing:
            raise ValueError(f"Krea 2 experiment configuration is missing: {', '.join(missing)}")
        return io.NodeOutput(*(configuration[name] for name in required), configuration.copy())


def _image_similarity(source, generated):
    source = source[:, :, :, :3].movedim(-1, 1)
    generated = generated[:, :, :, :3].movedim(-1, 1)
    if generated.shape[2:] != source.shape[2:]:
        generated = comfy.utils.common_upscale(generated, source.shape[3], source.shape[2], "area", "disabled")
    difference = generated - source
    mse = difference.square().mean().item()
    mae = difference.abs().mean().item()
    source_centered = source - source.mean()
    generated_centered = generated - generated.mean()
    cosine = torch.nn.functional.cosine_similarity(source_centered.flatten(), generated_centered.flatten(), dim=0).item()
    return {"mse": mse, "mae": mae, "psnr": -10.0 * math.log10(max(mse, 1e-12)), "pixel_cosine": cosine}


class Krea2ExperimentEvaluate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2ExperimentEvaluate",
            display_name="Evaluate Krea 2 Experiment",
            category="model/conditioning/qwen image",
            is_output_node=True,
            inputs=[
                io.Image.Input("source_image"),
                io.Image.Input("generated_image"),
                Krea2Experiment.Input("experiment"),
                Krea2Experiment.Input("configuration"),
                io.String.Input("results_file", default="krea2_visual_token_results.jsonl"),
            ],
            outputs=[io.String.Output(display_name="result")],
        )

    @classmethod
    def execute(cls, source_image, generated_image, experiment, configuration, results_file="krea2_visual_token_results.jsonl") -> io.NodeOutput:
        if source_image.shape[0] != 1 or generated_image.shape[0] != 1:
            raise ValueError("Krea 2 experiment evaluation requires one source and one generated image.")
        result = {"configuration": configuration, **experiment, **_image_similarity(source_image, generated_image)}
        result_json = json.dumps(result, sort_keys=True)
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        path = os.path.abspath(os.path.join(output_dir, results_file))
        if os.path.commonpath([output_dir, path]) != output_dir:
            raise ValueError("Experiment results file must be inside the ComfyUI output directory.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as output:
            output.write(result_json + "\n")
        return io.NodeOutput(result_json)


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
                    tooltip="Probability of selecting the first source. Remaining sources are selected with a checkerboard pattern.",
                ),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True, advanced=True, tooltip="Seed for the spatial-dither-random pattern."),
                io.Vae.Input("vae", optional=True),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, prompt, images: io.Autogrow.Type, fusion_method, block_size=2, dither_ratio=0.5, vae=None, seed=0) -> io.NodeOutput:
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
        conditioning = _fuse_conditionings(conditionings, tokens, visual_height, visual_width, fusion_method, block_size, dither_ratio, seed)

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
            TextEncodeKrea2VisualTokenControl,
            Krea2ExperimentConfiguration,
            Krea2ExperimentEvaluate,
            TextEncodeQwenImageEditPlus,
            TextEncodeQwenImageEditFusion,
            EmptyQwenImageLayeredLatentImage,
        ]


async def comfy_entrypoint() -> QwenExtension:
    return QwenExtension()
