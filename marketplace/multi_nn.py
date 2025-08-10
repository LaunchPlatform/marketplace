import typing

from tinygrad import nn
from tinygrad import Tensor


def repeat(w: Tensor, count: int) -> Tensor:
    return w.unsqueeze(0).repeat(count, *((1,) * len(w.shape))).contiguous()


class MultiModelBase:
    vendor_count: int

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError()


class MultiConv2d(MultiModelBase, nn.Conv2d):
    def __init__(
        self,
        vendor_count: int,
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
        self.vendor_count = vendor_count
        self.weight = repeat(self.weight, vendor_count)
        if self.bias is not None:
            self.bias = repeat(self.bias, vendor_count)

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        return x.conv2d(
            self.weight[i],
            self.bias[i] if self.bias is not None else None,
            self.groups,
            self.stride,
            self.dilation,
            self.padding,
        )


class MultiLinear(MultiModelBase, nn.Linear):
    def __init__(
        self, vendor_count: int, in_features: int, out_features: int, bias: bool = True
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.vendor_count = vendor_count
        self.weight = repeat(self.weight, vendor_count)
        if self.bias is not None:
            self.bias = repeat(self.bias, vendor_count)

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        return x.linear(self.weight[i].transpose(), self.bias[i])


class MultiModel(MultiModelBase):
    def __init__(
        self,
        layers: typing.List[MultiModelBase | typing.Callable[[Tensor], Tensor]],
        vendor_count: int | None = None,
    ):
        self.vendor_count = vendor_count
        self.layers: typing.List[MultiModelBase | typing.Callable[[Tensor], Tensor]] = (
            layers
        )
        for i, model in enumerate(self.layers):
            if not isinstance(model, MultiModelBase):
                continue
            if isinstance(model, SingletonModel):
                continue
            if self.vendor_count is None:
                self.vendor_count = model.vendor_count
            else:
                if model.vendor_count != self.vendor_count:
                    raise ValueError(
                        f"Layer {i} vendor count {model.vendor_count} does not match with "
                        f"the multi model's {self.vendor_count}"
                    )
        if self.vendor_count is None:
            self.vendor_count = 1

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        value = x
        for model in self.layers:
            if isinstance(model, MultiModelBase):
                value = model(i, value)
            else:
                value = model(value)
        return value


class SingletonModel(MultiModelBase):
    def __init__(self, model: typing.Callable[[Tensor], Tensor]):
        self.vendor_count = 1
        self.singleton = model

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        return self.singleton(x)
