import contextlib
import typing

from tinygrad import nn
from tinygrad import Tensor

from .random import RandomNumberGenerator


class DeltaModelBase:
    training: typing.ClassVar[bool] = False

    def __call__(self, rng: RandomNumberGenerator, x: Tensor) -> Tensor:
        raise NotImplementedError()

    class train(contextlib.ContextDecorator):
        def __init__(self, mode: bool = True):
            self.mode = mode

        def __enter__(self):
            self.prev = DeltaModelBase.training
            DeltaModelBase.training = self.mode

        def __exit__(self, exc_type, exc_value, traceback):
            DeltaModelBase.training = self.prev


class DeltaConv2d(DeltaModelBase, nn.Conv2d):
    def __call__(self, rng: RandomNumberGenerator, x: Tensor) -> Tensor:
        weight_delta = rng.uniform_like(self.weight)

        bias_delta = None
        if self.bias is not None:
            bias_delta = rng.uniform_like(self.bias)
        return x.conv2d(
            self.weight + weight_delta,
            self.bias + bias_delta if self.bias is not None else None,
            self.groups,
            self.stride,
            self.dilation,
            self.padding,
        )


class DeltaLinear(DeltaModelBase, nn.Linear):
    def __call__(self, rng: RandomNumberGenerator, x: Tensor) -> Tensor:
        weight_delta = rng.uniform_like(self.weight)

        bias_delta = None
        if self.bias is not None:
            bias_delta = rng.uniform_like(self.bias)

        return x.linear(
            (self.weight + weight_delta).transpose(),
            (self.bias + bias_delta) if self.bias is not None else None,
        )
