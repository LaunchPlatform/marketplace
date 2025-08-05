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
from marketplace.training import Spec
from marketplace.training import uniform_between


class Model:
    def __init__(self, layers: List[Callable[[Tensor], Tensor]]):
        self.layers: List[Callable[[Tensor], Tensor]] = layers

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

    VENDOR_COUNT = 16
    UPSTREAM_SAMPLING = 16
    OFFSPRING_COUNT = 8
    KEEP_COUNT = 8
    OFFSPRING_JITTER_SCALE = 0.1
    OFFSPRING_JITTER_OFFSET = 0.0001

    marketplace = [
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
        ),
        Spec(
            model_factory=lambda: Model(
                [
                    # nn.BatchNorm(32),
                    Tensor.max_pool2d
                ]
            ),
            vendor_count=VENDOR_COUNT,
            upstream_sampling=UPSTREAM_SAMPLING,
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
                    # nn.BatchNorm(64),
                    Tensor.max_pool2d,
                ]
            ),
            vendor_count=VENDOR_COUNT,
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
        Spec(
            model_factory=lambda: Model([lambda x: x.flatten(1), nn.Linear(576, 10)]),
            vendor_count=VENDOR_COUNT,
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
    ]

    for spec in marketplace:
        sample = spec.model_factory()
        params = nn.state.get_state_dict(sample)
        vendors = []
        for _ in range(spec.vendor_count):
            model = spec.model_factory()
            nn.state.load_state_dict(
                model, {key: params[key].clone() for key in params}, verbose=False
            )
            vendors.append(model)

        # spec.vendors = [sample for _ in range(spec.vendor_count)]
        # spec.vendors = [spec.model_factory()] * spec.vendor_count

    @TinyJit
    def train_step() -> tuple[Tensor, Tensor]:
        # samples = Tensor.randint(getenv("BS", 8), high=X_train.shape[0])
        samples = Tensor.arange(getenv("BS", 32))

        x = X_train[samples]
        y = Y_train[samples]

        output, paths = forward(marketplace, x)
        all_loss = Tensor.stack(
            *(logits.sparse_categorical_crossentropy(y) for logits in output)
        )
        return Tensor.stack(
            *(
                (
                    Tensor.zeros(len(marketplace), VENDOR_COUNT).scatter(
                        dim=1,
                        index=path.unsqueeze(1),
                        src=loss.neg().exp().repeat(VENDOR_COUNT, 1),
                    )
                )
                for loss, path in zip(all_loss, paths)
            ),
            dim=0,
        ).sum(axis=0), all_loss.min()

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

        all_loss = 0.0
        profit_matrix = Tensor.zeros(len(marketplace), VENDOR_COUNT)
        for _ in range(EVAL_CYCLE):
            profit_matrix_delta, loss = train_step()
            profit_matrix += profit_matrix_delta
            profit_matrix.realize()
            all_loss += loss.item()

        make_offsprings(
            profit_matrix=profit_matrix,
            marketplace=marketplace,
            offspring_count=OFFSPRING_COUNT,
            keep_count=KEEP_COUNT,
            jitter_scale=Tensor(OFFSPRING_JITTER_SCALE),
            jitter_offset=Tensor(OFFSPRING_JITTER_OFFSET),
        )

        end_time = time.perf_counter()
        run_time = end_time - start_time
        # if i % 10 == 9:
        #     test_acc = get_test_acc().item()
        t.set_description(
            f"loss: {all_loss / EVAL_CYCLE}, {GlobalCounters.global_ops * 1e-9 / run_time:9.2f} GFLOPS"
        )

    # print("profit matrix", profit_matrix.numpy())
    # # verify eval acc
    # if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    #     if test_acc >= target and test_acc != 100.0:
    #         print(colored(f"{test_acc=} >= {target}", "green"))
    #     else:
    #         raise ValueError(colored(f"{test_acc=} < {target}", "red"))
