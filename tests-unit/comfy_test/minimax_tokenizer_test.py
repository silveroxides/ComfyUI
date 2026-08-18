"""MiniMax H3 tokenizer regression tests."""

import torch
import pytest
from safetensors.torch import save_file

from comfy import sd1_clip
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer, VISION_END, VISION_START


def test_h3_prompt_embedding_uses_core_token_parser(tmp_path):
    embedding = torch.arange(2 * 5120, dtype=torch.float32).reshape(2, 5120)
    save_file({"qwen3vl_32b": embedding}, tmp_path / "h3_embedding.safetensors")
    tokenizer = MiniMaxH3Tokenizer(embedding_directory=str(tmp_path))

    entries = tokenizer.tokenize_with_weights("subject embedding:h3_embedding")["qwen3vl_32b"][0]
    tensors = [entry[0] for entry in entries if torch.is_tensor(entry[0])]

    assert len(tensors) == 2
    assert torch.equal(tensors[0], embedding[0])
    assert torch.equal(tensors[1], embedding[1])


def test_h3_keeps_explicit_vision_block_order():
    class Parser:
        def tokenize_with_weights(self, text, **_kwargs):
            return [[(ord(character), 1.0) for character in text]]

    tokenizer = object.__new__(MiniMaxH3Tokenizer)
    tokenizer.qwen3vl_32b = Parser()
    image = torch.zeros(1, 4, 4, 3)

    entries = tokenizer.tokenize_with_weights("prompt", images=[image])["qwen3vl_32b"][0]
    values = [entry[0] for entry in entries]
    label = [ord(character) for character in "<Picture 1>: "]

    assert values[:len(label)] == label
    assert values[len(label)] == VISION_START
    assert values[len(label) + 1]["type"] == "image"
    assert values[len(label) + 2] == VISION_END
    assert values[len(label) + 3:] == [ord(character) for character in "prompt"]


def test_h3_keeps_reference_media_order_and_video_timestamps():
    class Parser:
        calls = []

        def tokenize_with_weights(self, text, **_kwargs):
            self.calls.append(text)
            return [[(ord(character), 1.0) for character in text]]

    parser = Parser()
    tokenizer = object.__new__(MiniMaxH3Tokenizer)
    tokenizer.qwen3vl_32b = parser
    image = torch.zeros(1, 4, 4, 3)
    frames = torch.zeros(3, 4, 4, 3)

    entries = tokenizer.tokenize_with_weights(
        "prompt",
        minimax_ref_items=[
            {"type": "image", "data": image},
            {"type": "audio", "data": object()},
            {"type": "video", "data": frames, "timestamps": [0.0, 1.0, 2.0]},
        ],
    )["qwen3vl_32b"][0]

    vision_entries = [entry[0] for entry in entries if isinstance(entry[0], dict)]
    assert parser.calls == [
        "<Picture 1>: ",
        "<Audio 1>: ",
        "<Video 1>: ",
        "<0.5 seconds>",
        "<2.0 seconds>",
        "prompt",
    ]
    assert len(vision_entries) == 3
    assert vision_entries[0].get("minimax_video_block") is None
    assert all(entry["minimax_video_block"] for entry in vision_entries[1:])


def test_h3_embedding_rows_reach_process_tokens_without_image(tmp_path):
    class InputEmbeddings:
        def __call__(self, token_ids, out_dtype):
            return torch.empty((*token_ids.shape, 5120), dtype=out_dtype)

    class Transformer:
        @staticmethod
        def get_input_embeddings():
            return InputEmbeddings()

    embedding = torch.arange(2 * 5120, dtype=torch.float32).reshape(2, 5120)
    save_file({"qwen3vl_32b": embedding}, tmp_path / "h3_embedding.safetensors")
    tokenizer = MiniMaxH3Tokenizer(embedding_directory=str(tmp_path))
    entries = tokenizer.tokenize_with_weights("embedding:h3_embedding")["qwen3vl_32b"][0]

    clip_model = object.__new__(sd1_clip.SDClipModel)
    clip_model.special_tokens = {"pad": 151643}
    clip_model.transformer = Transformer()
    embeds, _, _, embeds_info = clip_model.process_tokens([[entry[0] for entry in entries]], "cpu")

    assert torch.equal(embeds[0], embedding)
    assert [(entry["type"], entry["index"], entry["size"]) for entry in embeds_info] == [
        ("embedding", 0, 1),
        ("embedding", 1, 1),
    ]


def test_h3_rejects_multiple_core_token_batches():
    class Parser:
        def tokenize_with_weights(self, _text, **_kwargs):
            return [[(1, 1.0)], [(2, 1.0)]]

    tokenizer = object.__new__(MiniMaxH3Tokenizer)
    tokenizer.qwen3vl_32b = Parser()

    with pytest.raises(ValueError, match="exceeds the supported prompt length"):
        tokenizer.tokenize_with_weights("prompt")
