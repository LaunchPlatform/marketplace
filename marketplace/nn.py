import typing

from tinygrad import Tensor


class Model:
    def __init__(self, *layers):
        self.layers: tuple[typing.Callable, ...] = layers

    def __call__(self, x: Tensor) -> Tensor:
        value = x
        for model in self.layers:
            value = model(value)
        return value
