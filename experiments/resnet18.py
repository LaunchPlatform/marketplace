from tinygrad import nn
from tinygrad import Tensor

from marketplace.multi_nn import MultiConv2d
from marketplace.multi_nn import MultiLinear
from marketplace.multi_nn import MultiModel
from marketplace.multi_nn import MultiModelBase
from marketplace.training import Spec


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
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        self.conv2 = MultiConv2d(
            vendor_count,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)

        self.downsample = lambda i, x: x
        if stride != 1 or in_channels != out_channels:
            self.downsample = MultiModel(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        out_channels, track_running_stats=False, affine=False
                    ),
                ]
            )

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        out = self.conv1(i, x)
        out = self.bn1(out)
        out = out.relu()
        out = self.conv2(i, out)
        out = self.bn2(out)
        out += self.downsample(i, x)
        out = out.relu()
        return out


def make_marketplace(num_classes: int = 10):
    return [
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(
                        4,
                        in_channels=3,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(64, track_running_stats=False, affine=False),
                    Tensor.relu,
                    lambda x: x.max_pool2d(
                        kernel_size=3, stride=2, padding=1, bias=False
                    ),
                ]
            ),
        ),
        # layer1
        Spec(
            model=MultiModel(
                [
                    BasicBlock(
                        8,
                        in_channels=64,
                        out_channels=64,
                        stride=1,
                    ),
                    BasicBlock(
                        8,
                        in_channels=64,
                        out_channels=64,
                        stride=1,
                    ),
                ]
            ),
            upstream_sampling=4,
        ),
        # layer2
        Spec(
            model=MultiModel(
                [
                    BasicBlock(
                        16,
                        in_channels=64,
                        out_channels=128,
                        stride=2,
                    ),
                    BasicBlock(
                        16,
                        in_channels=128,
                        out_channels=128,
                        stride=2,
                    ),
                ]
            ),
            upstream_sampling=8,
        ),
        # layer3
        Spec(
            model=MultiModel(
                [
                    BasicBlock(
                        32,
                        in_channels=128,
                        out_channels=256,
                        stride=2,
                    ),
                    BasicBlock(
                        32,
                        in_channels=256,
                        out_channels=256,
                        stride=2,
                    ),
                ]
            ),
            upstream_sampling=16,
        ),
        # layer4
        Spec(
            model=MultiModel(
                [
                    BasicBlock(
                        64,
                        in_channels=256,
                        out_channels=512,
                        stride=2,
                    ),
                    BasicBlock(
                        64,
                        in_channels=512,
                        out_channels=512,
                        stride=2,
                    ),
                    lambda x: x.avg_pool2d(kernel_size=7),
                    lambda x: x.flatten(1),
                ]
            ),
            upstream_sampling=32,
        ),
        # layer5
        Spec(
            model=MultiModel([MultiLinear(128, 512, num_classes)]),
            upstream_sampling=64,
        ),
    ]
