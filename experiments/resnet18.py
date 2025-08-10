from tinygrad import nn
from tinygrad import Tensor

from marketplace.multi_nn import MultiConv2d
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
                        kernel_size=3, stride=1, padding=1, bias=False
                    ),
                ]
            )
        )
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        # self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        # self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        # self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        # self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        #
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)
        #
        # def _make_layer(self, block, out_channels, num_blocks, stride):
        #     strides = [stride] + [1] * (num_blocks - 1)
        #     layers = []
        #     for stride in strides:
        #         layers.append(block(self.in_channels, out_channels, stride))
        #         self.in_channels = out_channels
        #     return nn.Sequential(*layers)
        #
        # def forward(self, x):
        #     out = self.conv1(x)
        #     out = self.bn1(out)
        #     out = self.relu(out)
        #     out = self.maxpool(out)
        #
        #     out = self.layer1(out)
        #     out = self.layer2(out)
        #     out = self.layer3(out)
        #     out = self.layer4(out)
        #
        #     out = self.avgpool(out)
        #     out = out.view(out.size(0), -1)
        #     out = self.fc(out)
        #     return out
    ]
