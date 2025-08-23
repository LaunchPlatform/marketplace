import contextlib
import typing

from tinygrad import nn
from tinygrad import Tensor
from tinygrad.nn import InstanceNorm

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


class DeltaModel(DeltaModelBase):
    def __init__(
        self,
        layers: typing.List[DeltaModelBase | typing.Callable[[Tensor], Tensor]],
    ):
        self.layers: typing.List[DeltaModelBase | typing.Callable[[Tensor], Tensor]] = (
            layers
        )

    def __call__(self, rng: RandomNumberGenerator, x: Tensor) -> Tensor:
        value = x
        for model in self.layers:
            if isinstance(model, DeltaModelBase):
                value = model(rng, value)
            else:
                value = model(value)
        return value


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


class DeltaInstanceNorm(DeltaModelBase, InstanceNorm):
    def __call__(self, rng: RandomNumberGenerator, x: Tensor) -> Tensor:
        x = (
            x.reshape(x.shape[0], self.num_features, -1)
            .layernorm(eps=self.eps)
            .reshape(x.shape)
        )
        if self.weight is None or self.bias is None:
            return x
        weight_delta = rng.uniform_like(self.weight)
        bias_delta = rng.uniform_like(self.bias)
        return x * (self.weight + weight_delta).reshape(1, -1, *[1] * (x.ndim - 2)) + (
            self.bias + bias_delta
        ).reshape(1, -1, *[1] * (x.ndim - 2))
