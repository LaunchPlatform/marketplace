import time

from tinygrad import nn
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import GlobalCounters
from tinygrad.helpers import trange

from marketplace.multi_nn import MultiConv2d
from marketplace.multi_nn import MultiLinear
from marketplace.multi_nn import MultiModel
from marketplace.multi_nn import MultiModelBase
from marketplace.training import forward
from marketplace.training import Spec


class BasicBlock(MultiModelBase):
    def __init__(
        self, vendor_count: int, in_channels: int, out_channels: int, stride: int = 1
    ):
        super().__init__()
        self.vendor_count = vendor_count
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


def make_marketplace(num_classes: int = 100):
    return [
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(
                        12,
                        in_channels=3,
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=False,
                    ),
                    nn.BatchNorm2d(64, track_running_stats=False, affine=False),
                    Tensor.relu,
                    lambda x: x.max_pool2d(
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ]
            ),
        ),
        # layer1
        Spec(
            model=MultiModel(
                [
                    BasicBlock(
                        16,
                        in_channels=64,
                        out_channels=64,
                        stride=1,
                    ),
                    BasicBlock(
                        16,
                        in_channels=64,
                        out_channels=64,
                        stride=1,
                    ),
                ]
            ),
            upstream_sampling=8,
        ),
        # layer2
        Spec(
            model=MultiModel(
                [
                    BasicBlock(
                        20,
                        in_channels=64,
                        out_channels=128,
                        stride=2,
                    ),
                    BasicBlock(
                        20,
                        in_channels=128,
                        out_channels=128,
                        stride=1,
                    ),
                ]
            ),
            upstream_sampling=10,
        ),
        # layer3
        Spec(
            model=MultiModel(
                [
                    BasicBlock(
                        24,
                        in_channels=128,
                        out_channels=256,
                        stride=2,
                    ),
                    BasicBlock(
                        24,
                        in_channels=256,
                        out_channels=256,
                        stride=1,
                    ),
                ]
            ),
            upstream_sampling=12,
        ),
        # layer4
        Spec(
            model=MultiModel(
                [
                    BasicBlock(
                        28,
                        in_channels=256,
                        out_channels=512,
                        stride=2,
                    ),
                    BasicBlock(
                        28,
                        in_channels=512,
                        out_channels=512,
                        stride=1,
                    ),
                    lambda x: x.avg_pool2d(kernel_size=7),
                    lambda x: x.flatten(1),
                ]
            ),
            upstream_sampling=14,
        ),
        # layer5
        Spec(
            model=MultiModel([MultiLinear(32, 512, num_classes)]),
            upstream_sampling=16,
        ),
    ]


def train(marketplace: list[Spec], step_count: int = 10, batch_size: int = 16):
    # X_train, Y_train, X_test, Y_test = mnist()
    # X_train = X_train.reshape(-1, 28, 28).astype(np.uint8)
    # X_test = X_test.reshape(-1, 28, 28).astype(np.uint8)
    # classes = 10

    @TinyJit
    def forward_step() -> tuple[Tensor, Tensor]:
        x = Tensor.randn(batch_size, 3, 224, 224)
        y = Tensor.randn(batch_size, 10)
        batch_logits, batch_paths = forward(marketplace, x)
        return Tensor.stack(
            *(logits.sparse_categorical_crossentropy(y) for logits in batch_logits),
            dim=0,
        ).realize(), batch_paths.realize()

    for i in (t := trange(step_count)):
        GlobalCounters.reset()

        start_time = time.perf_counter()

        batch_logits, batch_paths = forward_step()
        batch_logits.realize()
        batch_paths.realize()

        # print(batch_logits, batch_paths)
        # print(batch_logits.realize(), batch_paths.realize())
        end_time = time.perf_counter()
        run_time = end_time - start_time
        gflops = GlobalCounters.global_ops * 1e-9 / run_time

        loss = Tensor(0)
        current_forward_pass = 0
        lr = Tensor(0)
        test_acc = 0
        test_acc = 0

        t.set_description(
            f"loss: {loss.item():6.2f}, fw: {current_forward_pass}, rl: {lr.item():e}, "
            f"acc: {test_acc:5.2f}%, {gflops:9,.2f} GFLOPS"
        )


def main():
    train(make_marketplace(10))


if __name__ == "__main__":
    main()
