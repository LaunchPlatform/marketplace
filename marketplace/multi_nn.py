from tinygrad import nn
from tinygrad import Tensor


class MultiModel:
    replica: int

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError()


class MultiConv2d(MultiModel, nn.Conv2d):
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
