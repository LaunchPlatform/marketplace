import contextlib
import typing

from tinygrad import nn
from tinygrad import Tensor


class DeltaModelBase:
    training: typing.ClassVar[bool] = False
    vendor_count: int

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError()

    class train(contextlib.ContextDecorator):
        def __init__(self, mode: bool = True):
            self.mode = mode

        def __enter__(self):
            self.prev = DeltaModelBase.training
            DeltaModelBase.training = self.mode

        def __exit__(self, exc_type, exc_value, traceback):
            DeltaModelBase.training = self.prev


class DeltaLinear(DeltaModelBase, nn.Linear):
    def __init__(
        self, vendor_count: int, in_features: int, out_features: int, bias: bool = True
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.vendor_count = vendor_count

    def __call__(self, rng, x: Tensor) -> Tensor:
        weight_delta = rng.uniform_like(self.weight)

        bias_delta = None
        if self.bias is not None:
            bias_delta = rng.uniform_like(self.bias)

        return x.linear(
            (self.weight + weight_delta).transpose(),
            (self.bias + bias_delta) if self.bias is not None else None,
        )
