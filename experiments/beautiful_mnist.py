# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import itertools
import logging
import pathlib
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
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import safe_save

from marketplace.multi_nn import MultiConv2d
from marketplace.multi_nn import MultiLinear
from marketplace.multi_nn import MultiModel
from marketplace.training import forward
from marketplace.training import forward_with_path
from marketplace.training import mutate
from marketplace.training import Spec

logger = logging.getLogger(__name__)


def main():
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

    BATCH_SIZE = getenv("BS", 32)
    INITIAL_LEARNING_RATE = 1e-3
    LEARNING_RATE_DECAY_RATE = 1e-3
    MIN_DELTA = 1e-6
    PATIENT = 1
    MAX_FORWARD_PASS = 1024

    MARKETPLACE = [
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(4, 1, 32, 5),
                    Tensor.relu,
                ]
            ),
        ),
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(8, 32, 32, 5),
                    Tensor.relu,
                ]
            ),
            upstream_sampling=4,
            evolve=False,
        ),
        Spec(
            model=MultiModel(
                [nn.BatchNorm(32), Tensor.max_pool2d],
            ),
            evolve=False,
        ),
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(16, 32, 64, 3),
                    Tensor.relu,
                ]
            ),
            upstream_sampling=8,
        ),
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(32, 64, 64, 3),
                    Tensor.relu,
                ]
            ),
            upstream_sampling=16,
        ),
        Spec(
            model=MultiModel(
                [
                    nn.BatchNorm(64),
                    Tensor.max_pool2d,
                ]
            ),
            evolve=False,
        ),
        Spec(
            model=MultiModel([lambda x: x.flatten(1), MultiLinear(64, 576, 10)]),
            upstream_sampling=32,
        ),
    ]
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
    loss_history = []
    best_loss = float("inf")
    stall_counter = 0
    for i in (t := trange(getenv("STEPS", 100_000))):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing

        start_time = time.perf_counter()

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
        loss_history.append(loss.item())

        end_time = time.perf_counter()
        run_time = end_time - start_time
        learning_rate.replace(learning_rate * (1 - LEARNING_RATE_DECAY_RATE))
        if i % 10 == 9:
            test_acc = get_test_acc(path).item()
            writer.add_scalar("training/loss", loss.item(), i)
            writer.add_scalar("training/accuracy", test_acc, i)
            writer.add_scalar("training/forward_pass", current_forward_pass, i)
            writer.add_scalar("training/learning_rate", learning_rate.item(), i)
        if i % 1000 == 99:
            loss_mean = sum(loss_history) / len(loss_history)
            if loss_mean < best_loss - MIN_DELTA:
                best_loss = loss_mean
                stall_counter = 0
            else:
                stall_counter += 1
                if stall_counter > PATIENT:
                    stall_counter = 0
                    current_forward_pass += 1
                    current_forward_pass = min(current_forward_pass, MAX_FORWARD_PASS)
                    mutate_step.reset()
                    logger.info(
                        "Loss stall, increase forward pass to %s", current_forward_pass
                    )
            loss_history = []

            parameters = dict(
                itertools.chain.from_iterable(
                    [
                        (f"layer.{i}.{key}", weights[index])
                        for key, weights in get_state_dict(spec.model).items()
                    ]
                    for i, (index, spec) in enumerate(zip(path, MARKETPLACE))
                )
            )
            checkpoint_filepath = pathlib.Path(f"model-{i}.safetensors")
            checkpoint_tmp_filepath = checkpoint_filepath.with_suffix(".tmp")
            safe_save(
                parameters | dict(global_step=Tensor(i)), str(checkpoint_tmp_filepath)
            )
            checkpoint_tmp_filepath.rename(checkpoint_filepath)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
