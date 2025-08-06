import typing

from tinygrad import nn
from tinygrad import Tensor


class MultiModelBase:
    replica: int

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError()


class MultiConv2d(MultiModelBase, nn.Conv2d):
    def __init__(
        self,
        replica: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride=1,
        padding: int | tuple[int, ...] | str = 0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.replica = replica
        self.weight = self.weight.repeat(replica, *self.weight.shape)
        if self.bias is not None:
            self.bias = self.bias.repeat(replica, *self.bias.shape)

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        return x.conv2d(
            self.weight[i],
            self.bias[i],
            self.groups,
            self.stride,
            self.dilation,
            self.padding,
        )


class MultiLinear(MultiModelBase, nn.Linear):
    def __init__(self, replica: int, in_features: int, out_features: int, bias=True):
        super().__init__(in_features=in_features, out_features=out_features)
        self.replica = replica
        self.weight = self.weight.repeat(replica, *self.weight.shape)
        if self.bias is not None:
            self.bias = self.bias.repeat(replica, *self.bias.shape)

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        return x.linear(self.weight[i].transpose(), self.bias[i])


class MultiModel(MultiModelBase):
    def __init__(
        self, layers: typing.List[MultiModelBase | typing.Callable[[Tensor], Tensor]]
    ):
        self.layers: typing.List[MultiModelBase | typing.Callable[[Tensor], Tensor]] = (
            layers
        )

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        value = x
        for model in self.layers:
            if isinstance(model, MultiModelBase):
                value = model(i, value)
            else:
                value = model(value)
        return value
