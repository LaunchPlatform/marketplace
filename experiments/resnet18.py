from tinygrad import nn
from tinygrad import Tensor

from marketplace.multi_nn import MultiConv2d
from marketplace.multi_nn import MultiModel
from marketplace.multi_nn import MultiModelBase


class BasicBlock(MultiModelBase):
    def __init__(
        self, vendor_count: int, in_channels: int, out_channels: int, stride: int = 1
    ):
        super().__init__()
        self.conv1 = MultiConv2d(
            vendor_count,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.conv2 = MultiConv2d(
            vendor_count,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False)

        self.shortcut = lambda i, x: x
        if stride != 1 or in_channels != out_channels:
            self.shortcut = MultiModel(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels, track_running_stats=False),
                ]
            )

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        out = self.conv1(i, x)
        out = self.bn1(out)
        out = out.relu()
        out = self.conv2(i, out)
        out = self.bn2(out)
        out += self.shortcut(i, x)
        out = out.relu()
        return out
