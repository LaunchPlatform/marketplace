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
    BATCH_SIZE = getenv("BS", 32)
    BATCH_GROUP_SIZE = getenv("BGS", 16)
    INITIAL_LEARNING_RATE = 0.001

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
    learning_rate = Tensor(INITIAL_LEARNING_RATE)

    @TinyJit
    def train_step() -> tuple[Tensor, Tensor]:
        loss = []
        paths = []
        for _ in range(BATCH_GROUP_SIZE):
            samples = Tensor.randint(BATCH_SIZE, high=X_train.shape[0])

            x = X_train[samples]
            y = Y_train[samples]

            batch_logits, batch_paths = forward(MARKETPLACE, x)
            loss.append(
                Tensor.stack(
                    *(
                        logits.sparse_categorical_crossentropy(y)
                        for logits in batch_logits
                    ),
                    dim=0,
                ).realize()
            )
            paths.append(batch_paths.realize())

        combined_loss = Tensor.cat(*loss)
        combined_path = Tensor.cat(*paths)

        min_loss, min_loss_index = combined_loss.topk(1, largest=False)
        min_path = combined_path[min_loss_index].flatten()
        mutate(
            marketplace=MARKETPLACE,
            leading_path=min_path,
            jitter=learning_rate,
        )
        return min_loss.realize(), min_path.realize()

    @TinyJit
    def get_test_acc(path: Tensor) -> Tensor:
        return (
            forward_with_path(MARKETPLACE, X_test, path).argmax(axis=1) == Y_test
        ).mean() * 100

    test_acc = float("nan")
    for i in (t := trange(getenv("STEPS", 100000))):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        start_time = time.perf_counter()
        loss, path = train_step()
        end_time = time.perf_counter()
        run_time = end_time - start_time
        if i % 10 == 9:
            test_acc = get_test_acc(path).item()
        learning_rate.replace(learning_rate * (1 - 0.0001))
        t.set_description(
            f"loss: {loss.item():6.2f}, rl: {learning_rate.item():e}, acc: {test_acc:5.2f}%, {GlobalCounters.global_ops * 1e-9 / run_time:9,.2f} GFLOPS"
        )

    # verify eval acc
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))
