"""
Minimal DAC shim — copied from Voicebox backend/utils/dac_shim.py.

TADA only imports Snake1d. The real descript-audio-codec package pulls
onnx / tensorboard / matplotlib via descript-audiotools.
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


def install_dac_shim() -> None:
    try:
        import dac  # noqa: F401

        return
    except ImportError:
        pass

    dac_pkg = types.ModuleType("dac")
    dac_pkg.__path__ = []
    dac_pkg.__package__ = "dac"

    dac_nn = types.ModuleType("dac.nn")
    dac_nn.__path__ = []
    dac_nn.__package__ = "dac.nn"

    dac_nn_layers = types.ModuleType("dac.nn.layers")
    dac_nn_layers.__package__ = "dac.nn"
    dac_nn_layers.Snake1d = Snake1d
    dac_nn_layers.snake = snake

    dac_model = types.ModuleType("dac.model")
    dac_model.__path__ = []
    dac_model.__package__ = "dac.model"

    dac_model_dac = types.ModuleType("dac.model.dac")
    dac_model_dac.__package__ = "dac.model"
    dac_model_dac.Snake1d = Snake1d

    dac_pkg.nn = dac_nn
    dac_pkg.model = dac_model
    dac_nn.layers = dac_nn_layers
    dac_model.dac = dac_model_dac

    sys.modules["dac"] = dac_pkg
    sys.modules["dac.nn"] = dac_nn
    sys.modules["dac.nn.layers"] = dac_nn_layers
    sys.modules["dac.model"] = dac_model
    sys.modules["dac.model.dac"] = dac_model_dac
