"""Qwen3-VL chat template regression tests."""

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy import sd1_clip  # noqa: E402
from comfy.text_encoders import minimax, qwen3vl  # noqa: E402


PROMPT = "describe a cute anime girl with fennec ears"
TEXT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
IMAGE_TEMPLATE = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"


def build_template(monkeypatch, images=None, **kwargs):
    captured = {}

    def capture(self, text, return_word_ids=False, **kwargs):
        captured["text"] = text
        return {"qwen3vl_8b": [[]]}

    monkeypatch.setattr(sd1_clip.SD1Tokenizer, "tokenize_with_weights", capture)
    tokenizer = qwen3vl.Qwen3VLTokenizer.__new__(qwen3vl.Qwen3VLTokenizer)
    tokenizer.llama_template = TEXT_TEMPLATE
    tokenizer.llama_template_images = IMAGE_TEMPLATE
    tokenizer.tokenize_with_weights(PROMPT, images=images or [], **kwargs)
    return captured["text"]


def test_qwen3vl_instruct_text_template(monkeypatch):
    assert build_template(monkeypatch, thinking=False) == TEXT_TEMPLATE.format(PROMPT)


def test_qwen3vl_thinking_text_template(monkeypatch):
    assert build_template(monkeypatch, thinking=True) == TEXT_TEMPLATE.format(PROMPT) + "<think>\n"


def test_qwen3vl_instruct_image_template(monkeypatch):
    assert build_template(monkeypatch, images=[object()], thinking=False) == IMAGE_TEMPLATE.format(PROMPT)


def test_qwen3vl_thinking_image_template(monkeypatch):
    assert build_template(monkeypatch, images=[object()], thinking=True) == IMAGE_TEMPLATE.format(PROMPT) + "<think>\n"


@pytest.mark.parametrize("thinking", [False, True])
def test_qwen3vl_skip_template_passes_text_through(monkeypatch, thinking):
    assert build_template(monkeypatch, skip_template=True, thinking=thinking) == PROMPT


@pytest.mark.parametrize("thinking", [False, True])
def test_minimax_qwen3vl_32b_does_not_add_thinking_markers(thinking):
    tokenizer = minimax.MiniMaxH3Tokenizer.__new__(minimax.MiniMaxH3Tokenizer)
    tokenizer._text_ids = lambda text: [text]
    tokens = tokenizer.tokenize_with_weights(PROMPT, thinking=thinking)
    assert tokens == {"qwen3vl_32b": [[(PROMPT, 1.0)]]}
