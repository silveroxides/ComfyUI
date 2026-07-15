import asyncio

import pytest
import torch

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
if not torch.cuda.is_available():
    cli_args.cpu = True

try:
    from comfy_extras.nodes_qwen import KREA2_EXPERIMENT_CONFIG, Krea2ExperimentConfiguration, Krea2ExperimentEvaluate, QwenExtension, TextEncodeKrea2VisualTokenControl, TextEncodeQwenImageEditFusion, _flatten_images, _fuse_conditionings, _select_visual_tokens, _spatial_fusion_mask, _visual_token_indices, _visual_token_span
finally:
    cli_args.cpu = prior_cpu


def _tokens(image_position=1, suffix=1):
    pairs = [(1, 1.0)] * image_position
    pairs.append(({"type": "image", "data": torch.zeros(1, 32, 32, 3)}, 1.0))
    pairs.extend([(2, 1.0)] * suffix)
    return {"qwen3vl_4b": [pairs]}


def test_checkerboard_mask_multiple_sources():
    mask = _spatial_fusion_mask(2, 3, 3, "spatial-checkerboard", 2, 0.5, "cpu")
    assert mask.tolist() == [0, 1, 2, 1, 2, 0]


def test_block_interleave_mask():
    mask = _spatial_fusion_mask(4, 4, 2, "spatial-block-interleave", 2, 0.5, "cpu")
    assert mask.reshape(4, 4).tolist() == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ]


def test_dither_mask_honors_seed_and_two_source_ratio():
    first = _spatial_fusion_mask(4, 4, 2, "spatial-dither-random", 2, 0.5, "cpu", 7)
    second = _spatial_fusion_mask(4, 4, 2, "spatial-dither-random", 2, 0.5, "cpu", 7)
    changed = _spatial_fusion_mask(4, 4, 2, "spatial-dither-random", 2, 0.5, "cpu", 8)
    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    assert _spatial_fusion_mask(2, 2, 2, "spatial-dither-random", 2, 1.0, "cpu").tolist() == [0, 0, 0, 0]
    assert _spatial_fusion_mask(2, 2, 2, "spatial-dither-random", 2, 0.0, "cpu").tolist() == [1, 1, 1, 1]


def test_dither_ratio_selects_first_source_or_remaining_checkerboard():
    assert _spatial_fusion_mask(2, 3, 4, "spatial-dither-random", 2, 1.0, "cpu").tolist() == [0, 0, 0, 0, 0, 0]
    assert _spatial_fusion_mask(2, 3, 4, "spatial-dither-random", 2, 0.0, "cpu").tolist() == [1, 2, 3, 2, 3, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dither_mask_is_seeded_on_cuda():
    first = _spatial_fusion_mask(4, 4, 3, "spatial-dither-random", 2, 0.5, "cuda", 7)
    second = _spatial_fusion_mask(4, 4, 3, "spatial-dither-random", 2, 0.5, "cuda", 7)
    assert torch.equal(first, second)


def test_visual_span_accounts_for_stripped_prefix():
    tokens = _tokens(image_position=3, suffix=4)
    assert _visual_token_span(tokens, cond_length=9, visual_tokens=4) == (1, 5)


@pytest.mark.parametrize("ratio, expected", [(0.0, []), (0.25, [5, 7, 13, 15]), (1.0, list(range(16)))])
def test_uniform_visual_token_indices(ratio, expected):
    assert _visual_token_indices(4, 4, ratio, "uniform-grid") == expected


def test_legacy_tail_visual_token_indices():
    assert _visual_token_indices(2, 4, 0.25, "legacy-tail") == [6, 7]


def test_visual_token_ratio_boundaries():
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        _visual_token_indices(2, 4, -0.01, "uniform-grid")
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        _visual_token_indices(2, 4, 1.01, "uniform-grid")


def test_krea_visual_token_node_is_registered():
    nodes = asyncio.run(QwenExtension().get_node_list())
    assert TextEncodeKrea2VisualTokenControl in nodes
    assert Krea2ExperimentConfiguration in nodes
    assert Krea2ExperimentEvaluate in nodes


def test_krea_experiment_configuration_selects_index_and_returns_record():
    configurations = """[
        {"prompt": "first", "negative_prompt": "", "visual_resolution": 384, "visual_token_ratio": 0.0, "selection_method": "uniform-grid", "seed": 1, "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "width": 1024, "height": 1024},
        {"prompt": "second", "negative_prompt": "", "visual_resolution": 384, "visual_token_ratio": 0.0, "selection_method": "legacy-tail", "seed": 1, "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "width": 1024, "height": 1024}
    ]"""
    output = Krea2ExperimentConfiguration.execute(configurations, 1).args
    assert output[0] == "second"
    assert output[2:5] == (384, 0.0, "legacy-tail")
    assert output[-1] == 2


def test_select_visual_tokens_preserves_order_and_slices_attention_mask():
    conditioning = [[torch.arange(7).reshape(1, 7, 1), {"attention_mask": torch.ones(1, 7), "marker": True}]]
    selected = _select_visual_tokens(conditioning, _tokens(image_position=3, suffix=1), 4, [1, 3])
    cond, metadata = selected[0]

    assert cond.flatten().tolist() == [0, 1, 3, 5, 6]
    assert metadata["attention_mask"].shape == (1, 5)
    assert metadata["marker"] is True


def test_fusion_replaces_only_visual_tokens_and_preserves_dtype_and_metadata():
    tokens = [_tokens(), _tokens()]
    first = torch.tensor([[[10], [10], [10], [10], [10], [20]]], dtype=torch.float16)
    second = torch.tensor([[[30], [30], [30], [30], [30], [40]]], dtype=torch.float16)
    metadata = {"pooled_output": torch.tensor([1.0]), "marker": "first"}
    conditionings = [
        [[first, metadata]],
        [[second, {"pooled_output": torch.tensor([2.0])}]],
    ]

    fused = _fuse_conditionings(conditionings, tokens, 2, 2, "spatial-checkerboard", 2, 0.5)
    output, output_metadata = fused[0]

    assert output.dtype == torch.float16
    assert output.flatten().tolist() == [10, 10, 30, 30, 10, 20]
    assert output_metadata == metadata
    assert output_metadata is not metadata


def test_dither_seed_changes_fused_conditioning():
    tokens = [_tokens(), _tokens()]
    conditionings = [
        [[torch.zeros((1, 6, 1)), {}]],
        [[torch.ones((1, 6, 1)), {}]],
    ]

    first = _fuse_conditionings(conditionings, tokens, 2, 2, "spatial-dither-random", 2, 0.5, 7)[0][0]
    second = _fuse_conditionings(conditionings, tokens, 2, 2, "spatial-dither-random", 2, 0.5, 8)[0][0]

    assert not torch.equal(first, second)


def test_flatten_images_uses_numeric_input_order_and_splits_batches():
    images = {
        "image_10": torch.full((1, 2, 2, 3), 10.0),
        "image_2": torch.stack([torch.full((2, 2, 3), 2.0), torch.full((2, 2, 3), 3.0)]),
        "image_1": torch.full((1, 2, 2, 3), 1.0),
    }

    sources = _flatten_images(images)
    assert [source[0, 0, 0, 0].item() for source in sources] == [1.0, 2.0, 3.0, 10.0]


def test_node_uses_custom_krea_prompt_and_returns_fused_conditioning():
    class FakeClip:
        def tokenize(self, text, images):
            assert text.startswith("<|im_start|>system\nDescribe the image by detailing")
            assert "Picture 1:" not in text
            pairs = [
                (1, 1.0),
                ({"type": "image", "data": images[0]}, 1.0),
                (2, 1.0),
            ]
            return {"qwen3vl_4b": [pairs]}

        def encode_from_tokens_scheduled(self, tokens):
            image = next(pair[0]["data"] for pair in tokens["qwen3vl_4b"][0] if isinstance(pair[0], dict))
            value = image.mean()
            return [[torch.full((1, 146, 1), value, dtype=torch.float16), {"source": float(value)}]]

    result = TextEncodeQwenImageEditFusion.execute(
        FakeClip(),
        "test prompt",
        {"image_1": torch.zeros(1, 32, 32, 3), "image_2": torch.ones(1, 32, 32, 3)},
        "spatial-dither-random",
        seed=7,
    )
    changed_seed = TextEncodeQwenImageEditFusion.execute(
        FakeClip(),
        "test prompt",
        {"image_1": torch.zeros(1, 32, 32, 3), "image_2": torch.ones(1, 32, 32, 3)},
        "spatial-dither-random",
        seed=8,
    )
    conditioning = result.args[0]
    output, metadata = conditioning[0]

    assert output.dtype == torch.float16
    assert output.shape == (1, 146, 1)
    assert output[:, 0].item() == 0.0
    assert output[:, -1].item() == 0.0
    assert set(output[:, 1:-1].flatten().tolist()) == {0.0, 1.0}
    assert metadata == {"source": 0.0}
    assert not torch.equal(output, changed_seed.args[0][0][0])


def test_node_exposes_seed_control():
    inputs = {value.id: value for value in TextEncodeQwenImageEditFusion.define_schema().inputs}
    assert inputs["seed"].default == 0
    assert inputs["seed"].control_after_generate is True
