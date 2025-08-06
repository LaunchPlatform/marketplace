# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import time
from typing import Callable
from typing import List

from tinygrad import GlobalCounters
from tinygrad import nn
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import colored
from tinygrad.helpers import getenv
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist

from marketplace.training import forward
from marketplace.training import make_offsprings
from marketplace.training import mutate
from marketplace.training import Spec
from marketplace.training import uniform_between


class Model:
    def __init__(self, layers: List[Callable[[Tensor], Tensor]]):
        self.layers: List[Callable[[Tensor], Tensor]] = layers

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


class MultiConv2d(nn.Conv2d):
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


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

    VENDOR_COUNT = 32
    UPSTREAM_SAMPLING = 16
    OFFSPRING_JITTER_SCALE = 0.1
    OFFSPRING_JITTER_OFFSET = 0.001

    MARKETPLACE = [
        Spec(
            model_factory=lambda: Model(
                [
                    nn.Conv2d(1, 32, 5),
                    Tensor.relu,
                ]
            ),
            vendor_count=VENDOR_COUNT,
        ),
        Spec(
            model_factory=lambda: Model(
                [
                    nn.Conv2d(32, 32, 5),
                    Tensor.relu,
                ]
            ),
            vendor_count=VENDOR_COUNT,
            upstream_sampling=UPSTREAM_SAMPLING,
            evolve=False,
        ),
        Spec(
            model_factory=lambda: Model([nn.BatchNorm(32), Tensor.max_pool2d]),
            vendor_count=1,
            upstream_sampling=UPSTREAM_SAMPLING,
            evolve=False,
        ),
        Spec(
            model_factory=lambda: Model(
                [
                    nn.Conv2d(32, 64, 3),
                    Tensor.relu,
                ]
            ),
            vendor_count=VENDOR_COUNT,
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
        Spec(
            model_factory=lambda: Model(
                [
                    nn.Conv2d(64, 64, 3),
                    Tensor.relu,
                ]
            ),
            vendor_count=VENDOR_COUNT,
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
        Spec(
            model_factory=lambda: Model(
                [
                    nn.BatchNorm(64),
                    Tensor.max_pool2d,
                ]
            ),
            vendor_count=1,
            upstream_sampling=UPSTREAM_SAMPLING,
            evolve=False,
        ),
        Spec(
            model_factory=lambda: Model([lambda x: x.flatten(1), nn.Linear(576, 10)]),
            vendor_count=VENDOR_COUNT,
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
    ]

    for spec in MARKETPLACE:
        sample = spec.model_factory()
        params = nn.state.get_state_dict(sample)
        spec.vendors = []
        for _ in range(spec.vendor_count):
            model = spec.model_factory()
            nn.state.load_state_dict(
                model, {key: params[key].clone() for key in params}, verbose=False
            )
            spec.vendors.append(model)

        # spec.vendors = [sample for _ in range(spec.vendor_count)]
        # spec.vendors = [spec.model_factory()] * spec.vendor_count
    vendor_count_max = max([len(spec.vendors) for spec in MARKETPLACE])

    # @TinyJit
    def train_step() -> tuple[Tensor, Tensor]:
        samples = Tensor.randint(getenv("BS", 64), high=X_train.shape[0])
        # samples = Tensor.arange(64)

        x = X_train[samples]
        y = Y_train[samples]

        product_logits, paths = forward(MARKETPLACE, x)
        product_loss = Tensor.stack(
            *(logits.sparse_categorical_crossentropy(y) for logits in product_logits),
            dim=0,
        )
        min_loss, min_loss_index = product_loss.topk(1, largest=False)
        min_path = paths[min_loss_index].flatten()
        mutate(
            marketplace=MARKETPLACE,
            leading_path=min_path,
            jitter=Tensor(OFFSPRING_JITTER_OFFSET),
        )
        return min_loss.realize(), min_path.realize()

    #
    # @TinyJit
    # def get_test_acc() -> Tensor:
    #     return (model(X_test).argmax(axis=1) == Y_test).mean() * 100
    #
    EVAL_CYCLE = 16
    test_acc = float("nan")
    for i in (t := trange(getenv("STEPS", 1000))):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        start_time = time.perf_counter()

        # all_loss = 0.0
        # profit_matrix = Tensor.zeros(len(marketplace), VENDOR_COUNT)
        # for _ in range(EVAL_CYCLE):
        loss, path = train_step()

        end_time = time.perf_counter()
        run_time = end_time - start_time
        # if i % 10 == 9:
        #     test_acc = get_test_acc().item()
        t.set_description(
            f"loss: {loss.item()}, {GlobalCounters.global_ops * 1e-9 / run_time:9.2f} GFLOPS"
        )

    # print("profit matrix", profit_matrix.numpy())
    # # verify eval acc
    # if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    #     if test_acc >= target and test_acc != 100.0:
    #         print(colored(f"{test_acc=} >= {target}", "green"))
    #     else:
    #         raise ValueError(colored(f"{test_acc=} < {target}", "red"))
