# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import time

from tensorboardX import SummaryWriter
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
    FORWARD_PASS_SCHEDULE = [
        (0, 1),
        (100, 2),
        (500, 4),
        (1_000, 8),
        (5_000, 16),
        (10_000, 32),
        (20_000, 64),
        (40_000, 128),
    ]

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
    writer = SummaryWriter()

    @TinyJit
    def forward_step() -> tuple[Tensor, Tensor]:
        samples = Tensor.randint(BATCH_SIZE, high=X_train.shape[0])
        x = X_train[samples]
        y = Y_train[samples]
        batch_logits, batch_paths = forward(MARKETPLACE, x)
        return Tensor.stack(
            *(logits.sparse_categorical_crossentropy(y) for logits in batch_logits),
            dim=0,
        ).realize(), batch_paths.realize()

    @TinyJit
    def mutate_step(
        combined_loss: Tensor, combined_paths: Tensor
    ) -> tuple[Tensor, Tensor]:
        min_loss, min_loss_index = combined_loss.topk(1, largest=False)
        min_path = combined_paths[min_loss_index].flatten()
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
    current_forward_pass = 1
    for i in (t := trange(getenv("STEPS", 100_000))):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        start_time = time.perf_counter()

        for threshold, forward_pass in reversed(FORWARD_PASS_SCHEDULE):
            if i >= threshold:
                current_forward_pass = forward_pass
                break

        all_loss = []
        all_paths = []
        for _ in range(current_forward_pass):
            batch_loss, batch_path = forward_step()
            all_loss.append(batch_loss)
            all_paths.append(batch_path)

        combined_loss = Tensor.cat(*all_loss).realize()
        combined_paths = Tensor.cat(*all_paths).realize()
        loss, path = mutate_step(
            combined_loss=combined_loss, combined_paths=combined_paths
        )

        end_time = time.perf_counter()
        run_time = end_time - start_time
        if i % 10 == 9:
            test_acc = get_test_acc(path).item()
            writer.add_scalar("training/loss", loss.item(), i)
            writer.add_scalar("training/accuracy", test_acc, i)
            writer.add_scalar("training/forward_pass", current_forward_pass, i)
        learning_rate.replace(learning_rate * (1 - 0.0001))
        t.set_description(
            f"loss: {loss.item():6.2f}, fw: {current_forward_pass}, rl: {learning_rate.item():e}, "
            f"acc: {test_acc:5.2f}%, {GlobalCounters.global_ops * 1e-9 / run_time:9,.2f} GFLOPS"
        )

    # verify eval acc
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))
