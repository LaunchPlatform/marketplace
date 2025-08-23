import contextlib
import typing
from collections import OrderedDict
from typing import OrderedDict

from tinygrad import nn
from tinygrad import Tensor
from tinygrad.nn import InstanceNorm

from .random import RandomNumberGenerator


class ModelBase:
    training: typing.ClassVar[bool] = False

    def __call__(self, rng: RandomNumberGenerator, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def update(self, rng: RandomNumberGenerator):
        raise NotImplementedError()

    class train(contextlib.ContextDecorator):
        def __init__(self, mode: bool = True):
            self.mode = mode

        def __enter__(self):
            self.prev = ModelBase.training
            ModelBase.training = self.mode

        def __exit__(self, exc_type, exc_value, traceback):
            ModelBase.training = self.prev


class Model(ModelBase):
    def __init__(
        self,
        layers: typing.List[ModelBase | typing.Callable[[Tensor], Tensor]],
    ):
        self.layers: typing.List[ModelBase | typing.Callable[[Tensor], Tensor]] = layers

    def __call__(self, rng: RandomNumberGenerator, x: Tensor) -> Tensor:
        value = x
        for model in self.layers:
            if isinstance(model, ModelBase):
                value = model(rng, value)
            else:
                value = model(value)
        return value

    def update(self, rng: RandomNumberGenerator) -> OrderedDict[str, Tensor]:
        params = OrderedDict()
        for i, model in enumerate(self.layers):
            if not isinstance(model, ModelBase):
                continue
            model_params = model.update(rng)
            params.update(
                OrderedDict(
                    [
                        (f"layers[{i}].{key}", value)
                        for key, value in model_params.items()
                    ]
                )
            )
        return params


class Conv2D(ModelBase, nn.Conv2d):
    def __call__(self, rng: RandomNumberGenerator, x: Tensor) -> Tensor:
        weight_delta = rng.delta_like(self.weight)

        bias_delta = None
        if self.bias is not None:
            bias_delta = rng.delta_like(self.bias)
        return x.conv2d(
            self.weight + weight_delta,
            self.bias + bias_delta if self.bias is not None else None,
            self.groups,
            self.stride,
            self.dilation,
            self.padding,
        )

    def update(self, rng: RandomNumberGenerator) -> OrderedDict[str, Tensor]:
        params = OrderedDict()
        params["weight"] = self.weight.assign(self.weight + rng.delta_like(self.weight))
        if self.bias is not None:
            params["bias"] = self.bias.assign(self.bias + rng.delta_like(self.bias))
        return params


class Linear(ModelBase, nn.Linear):
    def __call__(self, rng: RandomNumberGenerator, x: Tensor) -> Tensor:
        weight_delta = rng.delta_like(self.weight)

        bias_delta = None
        if self.bias is not None:
            bias_delta = rng.delta_like(self.bias)

        return x.linear(
            (self.weight + weight_delta).transpose(),
            (self.bias + bias_delta) if self.bias is not None else None,
        )

    def update(self, rng: RandomNumberGenerator) -> OrderedDict[str, Tensor]:
        params = OrderedDict()
        params["weight"] = self.weight.assign(self.weight + rng.delta_like(self.weight))
        if self.bias is not None:
            params["bias "] = self.bias.assign(self.bias + rng.delta_like(self.bias))
        return params


class InstanceNorm(ModelBase, InstanceNorm):
    def __call__(self, rng: RandomNumberGenerator, x: Tensor) -> Tensor:
        x = (
            x.reshape(x.shape[0], self.num_features, -1)
            .layernorm(eps=self.eps)
            .reshape(x.shape)
        )
        if self.weight is None or self.bias is None:
            return x
        weight_delta = rng.delta_like(self.weight)
        bias_delta = rng.delta_like(self.bias)
        return x * (self.weight + weight_delta).reshape(1, -1, *[1] * (x.ndim - 2)) + (
            self.bias + bias_delta
        ).reshape(1, -1, *[1] * (x.ndim - 2))

    def update(self, rng: RandomNumberGenerator) -> OrderedDict[str, Tensor]:
        params = OrderedDict()
        if self.weight is None or self.bias is None:
            return params
        params["weight"] = self.weight.assign(self.weight + rng.delta_like(self.weight))
        params["bias"] = self.bias.assign(self.bias + rng.delta_like(self.bias))
        return params
