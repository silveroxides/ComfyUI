import pytest
import torch

import comfy.cli_args
import comfy.model_management
from comfy.model_patcher import ModelPatcher


@pytest.mark.parametrize(
    ("flags", "enabled", "disabled"),
    (([], False, False), (["--fast-disk"], True, False), (["--disable-fast-disk"], False, True)),
)
def test_fast_disk_arguments(flags, enabled, disabled):
    args = comfy.cli_args.parser.parse_args(flags)
    assert args.fast_disk is enabled
    assert args.disable_fast_disk is disabled


@pytest.mark.parametrize(
    "flags",
    (["--fast-disk", "--disable-fast-disk"], ["--disable-fast-disk", "--fast-disk"]),
)
def test_fast_disk_arguments_are_mutually_exclusive(flags, capsys):
    with pytest.raises(SystemExit) as exc:
        comfy.cli_args.parser.parse_args(flags)
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flags", "detected", "expected"),
    (
        ([], True, True),
        ([], False, False),
        ([], None, False),
        (["--fast-disk"], True, True),
        (["--fast-disk"], False, True),
        (["--fast-disk"], None, True),
        (["--disable-fast-disk"], True, False),
        (["--disable-fast-disk"], False, False),
        (["--disable-fast-disk"], None, False),
    ),
)
def test_fast_disk_policy_and_clone(monkeypatch, flags, detected, expected):
    args = comfy.cli_args.parser.parse_args(flags)
    monkeypatch.setattr(comfy.model_management.args, "fast_disk", args.fast_disk)
    monkeypatch.setattr(comfy.model_management.args, "disable_fast_disk", args.disable_fast_disk)
    device = torch.device("cpu")
    patcher = ModelPatcher(torch.nn.Linear(2, 2), device, device, fast_disk=detected)

    assert patcher.fast_disk is expected
    assert patcher.clone().fast_disk is expected
