import torch

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
if not torch.cuda.is_available():
    cli_args.cpu = True

try:
    from comfy_extras.nodes_qwen import TextEncodeQwenImageEditFusion, _flatten_images, _fuse_conditionings, _spatial_fusion_mask, _visual_token_span
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


def test_dither_mask_is_deterministic_and_honors_two_source_ratio():
    first = _spatial_fusion_mask(4, 4, 2, "spatial-dither-random", 2, 0.5, "cpu")
    second = _spatial_fusion_mask(4, 4, 2, "spatial-dither-random", 2, 0.5, "cpu")
    assert torch.equal(first, second)
    assert _spatial_fusion_mask(2, 2, 2, "spatial-dither-random", 2, 1.0, "cpu").tolist() == [0, 0, 0, 0]
    assert _spatial_fusion_mask(2, 2, 2, "spatial-dither-random", 2, 0.0, "cpu").tolist() == [1, 1, 1, 1]


def test_visual_span_accounts_for_stripped_prefix():
    tokens = _tokens(image_position=3, suffix=4)
    assert _visual_token_span(tokens, cond_length=9, visual_tokens=4) == (1, 5)


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
        "spatial-checkerboard",
    )
    conditioning = result.args[0]
    output, metadata = conditioning[0]

    assert output.dtype == torch.float16
    assert output.shape == (1, 146, 1)
    assert output[:, 0].item() == 0.0
    assert output[:, -1].item() == 0.0
    assert set(output[:, 1:-1].flatten().tolist()) == {0.0, 1.0}
    assert metadata == {"source": 0.0}
