import re

import pytest
import torch

from comfy.text_encoders.ideogram4 import Ideogram4Qwen3VLTokenizer
from comfy.text_encoders.krea2 import Krea2Tokenizer
from comfy.text_encoders.qwen3vl import Qwen3VLTokenizer
from comfy_extras.nodes_cond import CLIPTextEncodeImagePlaceholders


IMAGE_INPUT_PATTERN = re.compile(r"\bimage_input_(\d+)\b", re.IGNORECASE)


def _image_values(tokens):
    return [float(pair[0]["data"].mean()) for rows in tokens.values() for row in rows for pair in row if isinstance(pair[0], dict)]


def test_qwen3vl_inline_images_follow_text_order_expand_batches_and_repeat():
    tokenizer = Qwen3VLTokenizer()
    first = torch.stack([torch.full((32, 32, 3), 1.0), torch.full((32, 32, 3), 2.0)])
    second = torch.full((1, 32, 32, 3), 3.0)

    tokens = tokenizer.tokenize_with_weights(
        'image_input_2 {"first": "IMAGE_INPUT_1", "again": "image_input_1"}',
        inline_images={"image_input_1": first, "image_input_2": second},
    )

    assert _image_values(tokens) == [3.0, 1.0, 2.0, 1.0, 2.0]


@pytest.mark.parametrize("tokenizer_class", [Ideogram4Qwen3VLTokenizer, Krea2Tokenizer])
def test_qwen3vl_subclasses_use_inline_images_without_prepending_an_extra_image(tokenizer_class):
    tokenizer = tokenizer_class()
    image = torch.ones(1, 32, 32, 3)

    tokens = tokenizer.tokenize_with_weights('{"desc": "image_input_1, red coat"}', inline_images={"image_input_1": image})

    assert _image_values(tokens) == [1.0]


def test_krea_inline_image_remains_after_the_stripped_template_prefix():
    tokenizer = Krea2Tokenizer()
    tokens = tokenizer.tokenize_with_weights("image_input_1 test prompt", inline_images={"image_input_1": torch.ones(1, 32, 32, 3)})
    pairs = tokens["qwen3vl_4b"][0]
    im_start_positions = [i for i, pair in enumerate(pairs) if pair[0] == 151644]
    image_position = next(i for i, pair in enumerate(pairs) if isinstance(pair[0], dict))

    assert im_start_positions[1] < image_position


def test_qwen3vl_ordinary_image_tokenization_is_unchanged():
    tokenizer = Qwen3VLTokenizer()
    images = [torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3)]

    tokens = tokenizer.tokenize_with_weights("test prompt", images=images)

    assert _image_values(tokens) == [0.0, 1.0]


def test_qwen3vl_inline_images_reject_invalid_calls():
    tokenizer = Qwen3VLTokenizer()
    image = torch.zeros(1, 32, 32, 3)

    with pytest.raises(ValueError, match="cannot be combined"):
        tokenizer.tokenize_with_weights("image_input_1", images=[image], inline_images={"image_input_1": image})
    with pytest.raises(ValueError, match="No image was provided for image_input_2"):
        tokenizer.tokenize_with_weights("image_input_2", inline_images={"image_input_1": image})
    with pytest.raises(ValueError, match="No inline image placeholders"):
        tokenizer.tokenize_with_weights("test prompt", inline_images={"image_input_1": image})
    with pytest.raises(ValueError, match="Duplicate inline image placeholder"):
        tokenizer.tokenize_with_weights("image_input_1", inline_images={"image_input_1": image, "IMAGE_INPUT_1": image})


class FakeClip:
    def __init__(self, supports_inline=True):
        self.supports_inline = supports_inline
        self.inline_images = None
        self.encoded = False

    def tokenize(self, text, inline_images):
        self.inline_images = inline_images
        pairs = [(1, 1.0)]
        if self.supports_inline:
            for match in IMAGE_INPUT_PATTERN.finditer(text):
                image = inline_images[match.group(0).lower()]
                pairs.extend(({"type": "image", "data": image[i:i + 1]}, 1.0) for i in range(image.shape[0]))
        return {"qwen3vl_8b": [pairs]}

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded = True
        return [[torch.zeros(1, 1, 1), {"images": _image_values(tokens)}]]


def test_placeholder_node_preserves_port_numbers_expands_batches_and_ignores_unused_inputs():
    clip = FakeClip()
    images = {
        "image_1": torch.full((1, 8, 8, 4), 1.0),
        "image_2": torch.stack([torch.full((8, 8, 4), 2.0), torch.full((8, 8, 4), 3.0)]),
        "image_3": torch.full((1, 8, 8, 4), 4.0),
    }

    result = CLIPTextEncodeImagePlaceholders.execute(clip, "image_input_2 IMAGE_INPUT_1 image_input_2", images)

    assert result.args[0][0][1]["images"] == [2.0, 3.0, 1.0, 2.0, 3.0]
    assert clip.inline_images["image_input_1"].shape == (1, 8, 8, 3)
    assert clip.inline_images["image_input_2"].shape == (2, 8, 8, 3)
    assert clip.encoded


def test_placeholder_node_rejects_missing_placeholders_and_unsupported_tokenizers():
    image = torch.zeros(1, 8, 8, 3)

    with pytest.raises(ValueError, match="requires at least one"):
        CLIPTextEncodeImagePlaceholders.execute(FakeClip(), "test prompt", {"image_1": image})
    with pytest.raises(ValueError, match="No image is connected for image_input_2"):
        CLIPTextEncodeImagePlaceholders.execute(FakeClip(), "image_input_2", {"image_1": image})
    with pytest.raises(ValueError, match="does not support inline images"):
        CLIPTextEncodeImagePlaceholders.execute(FakeClip(supports_inline=False), "image_input_1", {"image_1": image})


def test_placeholder_node_limits_total_pixels_across_repeated_images():
    image = torch.zeros(1, 600, 600, 3)

    with pytest.raises(ValueError, match="1080000 total pixels"):
        CLIPTextEncodeImagePlaceholders.execute(FakeClip(), "image_input_1 image_input_1 image_input_1", {"image_1": image})

    result = CLIPTextEncodeImagePlaceholders.execute(
        FakeClip(),
        "image_input_1 image_input_1 image_input_1",
        {"image_1": image},
        max_total_pixels=1080000,
    )
    assert result.args[0][0][1]["images"] == [0.0, 0.0, 0.0]


def test_placeholder_node_schema():
    schema = CLIPTextEncodeImagePlaceholders.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "CLIPTextEncodeImagePlaceholders"
    assert schema.category == "model/conditioning"
    assert "text" in inputs
    assert "images" in inputs
    assert inputs["max_total_pixels"].default == 1024 * 1024
