from typing_extensions import override
from typing import Callable

import torch

import comfy.model_management
from comfy_api.latest import ComfyExtension, io

import nodes

class EmptyChromaRadianceLatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmptyChromaRadianceLatentImage",
            category="latent/chroma_radiance",
            inputs=[
                io.Int.Input(id="width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input(id="height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input(id="batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent().Output()],
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int=1) -> io.NodeOutput:
        latent = torch.zeros((batch_size, 3, height, width), device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples":latent})


class ChromaRadianceOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChromaRadianceOptions",
            category="model_patches/chroma_radiance",
            description="Allows setting advanced options for the Chroma Radiance model.",
            inputs=[
                io.Model.Input(id="model"),
                io.Boolean.Input(
                    id="preserve_wrapper",
                    default=True,
                    tooltip="When enabled, will delegate to an existing model function wrapper if it exists. Generally should be left enabled.",
                ),
                io.Float.Input(
                    id="start_sigma",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    tooltip="First sigma that these options will be in effect.",
                ),
                io.Float.Input(
                    id="end_sigma",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    tooltip="Last sigma that these options will be in effect.",
                ),
                io.Int.Input(
                    id="nerf_tile_size",
                    default=-1,
                    min=-1,
                    tooltip="Allows overriding the default NeRF tile size. -1 means use the default (32). 0 means use non-tiling mode (may require a lot of VRAM).",
                ),
                io.Boolean.Input(
                    id="grid_mitigation",
                    default=False,
                    tooltip="When enabled, will attempt to mitigate the grid artifacts when generating.",
                ),
                # === CORRECTED INPUT START ===
                io.Float.Input(
                    id="mitigation_start_sigma",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Grid mitigation will only activate when the noise sigma is BELOW this value. (e.g., 0.5 activates for the last half of sampling)",
                ),
                # === CORRECTED INPUT END ===
                io.Int.Input(
                    id="num_offsets",
                    default=1,
                    min=1,
                    max=16,
                    tooltip="Number of random crops to blend for grid mitigation. Higher values give better quality but are slower.",
                ),
                io.Int.Input(
                    id="offset_size",
                    default=15,
                    min=1,
                    max=15,
                    tooltip="Maximum pixel shift for grid mitigation crops. Default of 15 is recommended for 16x16 patches.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        *,
        model: io.Model.Type,
        preserve_wrapper: bool,
        start_sigma: float,
        end_sigma: float,
        nerf_tile_size: int,
        grid_mitigation: bool,
        mitigation_start_sigma: float,
        num_offsets: int,
        offset_size: int,
    ) -> io.NodeOutput:
        radiance_options = {}
        if nerf_tile_size >= 0:
            radiance_options["nerf_tile_size"] = nerf_tile_size

        if grid_mitigation:
            radiance_options["grid_mitigation_enabled"] = grid_mitigation
            radiance_options["mitigation_start_sigma"] = mitigation_start_sigma
            radiance_options["num_offsets"] = num_offsets
            radiance_options["offset_size"] = offset_size

        if not radiance_options:
            return io.NodeOutput(model)

        old_wrapper = model.model_options.get("model_function_wrapper")

        def model_function_wrapper(apply_model: Callable, args: dict) -> torch.Tensor:
            c = args["c"].copy()
            sigma = args["timestep"].max().detach().cpu().item()
            if end_sigma <= sigma <= start_sigma:
                transformer_options = c.get("transformer_options", {}).copy()
                existing_opts = transformer_options.get("chroma_radiance_options", {}).copy()
                existing_opts.update(radiance_options)
                transformer_options["chroma_radiance_options"] = existing_opts
                c["transformer_options"] = transformer_options
            if not (preserve_wrapper and old_wrapper):
                return apply_model(args["input"], args["timestep"], **c)
            return old_wrapper(apply_model, args | {"c": c})

        model = model.clone()
        model.set_model_unet_function_wrapper(model_function_wrapper)
        return io.NodeOutput(model)


class ChromaRadianceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyChromaRadianceLatentImage,
            ChromaRadianceOptions,
        ]


async def comfy_entrypoint() -> ChromaRadianceExtension:
    return ChromaRadianceExtension()