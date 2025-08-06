# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import time

from tinygrad import GlobalCounters
from tinygrad import nn
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import colored
from tinygrad.helpers import getenv
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist

from .multi_nn import MultiConv2d
from .multi_nn import MultiLinear
from .multi_nn import MultiModel
from .training import forward
from .training import forward_with_path
from .training import mutate
from .training import Spec


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

    VENDOR_COUNT = 32
    UPSTREAM_SAMPLING = 16
    OFFSPRING_JITTER_SCALE = 0.1
    OFFSPRING_JITTER_OFFSET = 0.001

    MARKETPLACE = [
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(VENDOR_COUNT, 1, 32, 5),
                    Tensor.relu,
                ]
            ),
        ),
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(VENDOR_COUNT, 32, 32, 5),
                    Tensor.relu,
                ]
            ),
            upstream_sampling=UPSTREAM_SAMPLING,
            evolve=False,
        ),
        Spec(
            model=MultiModel(
                [nn.BatchNorm(32), Tensor.max_pool2d],
            ),
            upstream_sampling=UPSTREAM_SAMPLING,
            evolve=False,
        ),
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(VENDOR_COUNT, 32, 64, 3),
                    Tensor.relu,
                ]
            ),
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(VENDOR_COUNT, 64, 64, 3),
                    Tensor.relu,
                ]
            ),
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
        Spec(
            model=MultiModel(
                [
                    nn.BatchNorm(64),
                    Tensor.max_pool2d,
                ]
            ),
            upstream_sampling=UPSTREAM_SAMPLING,
            evolve=False,
        ),
        Spec(
            model=MultiModel(
                [lambda x: x.flatten(1), MultiLinear(VENDOR_COUNT, 576, 10)]
            ),
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
    ]
    max_vendor_count = max([spec.model.vendor_count for spec in MARKETPLACE])

    @TinyJit
    def train_step() -> tuple[Tensor, Tensor]:
        samples = Tensor.randint(getenv("BS", 64), high=X_train.shape[0])

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
    @TinyJit
    def get_test_acc(path: Tensor) -> Tensor:
        return (
            forward_with_path(MARKETPLACE, X_test, path).argmax(axis=1) == Y_test
        ).mean() * 100

    test_acc = float("nan")
    for i in (t := trange(getenv("STEPS", 10000))):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        start_time = time.perf_counter()
        loss, path = train_step()
        end_time = time.perf_counter()
        run_time = end_time - start_time
        if i % 10 == 9:
            test_acc = get_test_acc(path).item()
        t.set_description(
            f"loss: {loss.item():6.2f}, acc: {test_acc:5.2f}%, {GlobalCounters.global_ops * 1e-9 / run_time:9,.2f} GFLOPS"
        )

    # verify eval acc
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))
