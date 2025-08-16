# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import functools
import logging
import pathlib
import time

import click
import mlflow
from tinygrad import GlobalCounters
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import getenv
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist

from .utils import ensure_experiment
from marketplace.multi_nn import MultiBatchNorm
from marketplace.multi_nn import MultiConv2d
from marketplace.multi_nn import MultiLinear
from marketplace.multi_nn import MultiModel
from marketplace.multi_nn import MultiModelBase
from marketplace.training import forward
from marketplace.training import forward_with_path
from marketplace.training import mutate
from marketplace.training import Spec
from marketplace.utils import write_checkpoint

logger = logging.getLogger(__name__)


@functools.cache
def load_data():
    return mnist(fashion=getenv("FASHION"))


def make_marketplace(
    structure: list[tuple[int, int]] | None = None,
    default_vendor_count: int = 8,
):
    if structure is None:
        structure = [
            # layer 0
            (default_vendor_count, 0),
            # layer 1
            (default_vendor_count, 0),
            # layer 2
            (default_vendor_count, 0),
        ]
    l0_vendor_count = structure[0][0]
    l1_vendor_count = structure[1][0]
    l1_upstream_sampling = structure[1][1]
    l2_vendor_count = structure[2][0]
    l2_upstream_sampling = structure[2][1]
    return [
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(l0_vendor_count, 1, 32, 5),
                    Tensor.relu,
                    MultiConv2d(l0_vendor_count, 32, 32, 5),
                    Tensor.relu,
                    MultiBatchNorm(l0_vendor_count, 32),
                    Tensor.max_pool2d,
                ]
            )
        ),
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(l1_vendor_count, 32, 64, 3),
                    Tensor.relu,
                    MultiConv2d(l1_vendor_count, 64, 64, 3),
                    Tensor.relu,
                    MultiBatchNorm(l1_vendor_count, 64),
                    Tensor.max_pool2d,
                    lambda x: x.flatten(1),
                ]
            ),
            upstream_sampling=l1_upstream_sampling,
        ),
        Spec(
            model=MultiModel([MultiLinear(l2_vendor_count, 576, 10)]),
            upstream_sampling=l2_upstream_sampling,
        ),
    ]


def make_deep_marketplace(
    structure: list[tuple[int, int]] | None = None,
    default_vendor_count: int = 4,
):
    if structure is None:
        structure = [(default_vendor_count, 0)] * 5
    return [
        # layer0
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(structure[0][0], 1, 32, 5),
                    Tensor.relu,
                ]
            )
        ),
        # layer1
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(structure[1][0], 32, 32, 5),
                    Tensor.relu,
                    MultiBatchNorm(structure[2][0], 32),
                    Tensor.max_pool2d,
                ]
            ),
            upstream_sampling=structure[1][1],
        ),
        # layer2
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(structure[2][0], 32, 64, 3),
                    Tensor.relu,
                ]
            ),
            upstream_sampling=structure[2][1],
        ),
        # layer3
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(structure[3][0], 64, 64, 3),
                    Tensor.relu,
                    MultiBatchNorm(structure[3][0], 64),
                    Tensor.max_pool2d,
                    lambda x: x.flatten(1),
                ]
            ),
            upstream_sampling=structure[3][1],
        ),
        # layer4
        Spec(
            model=MultiModel([MultiLinear(structure[4][0], 576, 10)]),
            upstream_sampling=structure[4][1],
        ),
    ]


def make_marketplace_without_cross_mixing(vendor_count: int):
    return [
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(vendor_count, 1, 32, 5),
                    Tensor.relu,
                    MultiConv2d(vendor_count, 32, 32, 5),
                    Tensor.relu,
                    MultiBatchNorm(vendor_count, 32),
                    Tensor.max_pool2d,
                    MultiConv2d(vendor_count, 32, 64, 3),
                    Tensor.relu,
                    MultiConv2d(vendor_count, 64, 64, 3),
                    Tensor.relu,
                    MultiBatchNorm(vendor_count, 64),
                    Tensor.max_pool2d,
                    lambda x: x.flatten(1),
                    MultiLinear(vendor_count, 576, 10),
                ]
            ),
        ),
    ]


def train(
    step_count: int,
    batch_size: int,
    initial_lr: float,
    lr_decay_rate: float,
    marketplace: list[Spec],
    initial_forward_pass: int = 1,
    metrics_per_steps: int = 10,
    forward_pass_schedule: list[tuple[int, int]] | None = None,
    checkpoint_filepath: pathlib.Path | None = None,
    checkpoint_per_steps: int = 1000,
):
    logger.info(
        "Running beautiful MNIST with step_count=%s, batch_size=%s, init_lr=%s, lr_decay=%s, "
        "initial_forward_pass=%s, metrics_per_steps=%s, forward_pass_schedule=%s, "
        "checkpoint_filepath=%s, checkpoint_per_steps=%s",
        step_count,
        batch_size,
        initial_lr,
        lr_decay_rate,
        initial_forward_pass,
        metrics_per_steps,
        forward_pass_schedule,
        checkpoint_filepath,
        checkpoint_per_steps,
    )

    lr = Tensor(initial_lr)

    mlflow.log_param("step_count", step_count)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("initial_forward_pass", initial_forward_pass)
    mlflow.log_param("lr", initial_lr)
    mlflow.log_param("lr_decay_rate", lr_decay_rate)
    mlflow.log_param("forward_pass_schedule", forward_pass_schedule)
    mlflow.log_param("metrics_per_steps", metrics_per_steps)
    mlflow.log_param("checkpoint_per_steps", checkpoint_per_steps)

    X_train, Y_train, X_test, Y_test = load_data()

    @TinyJit
    def forward_step() -> tuple[Tensor, Tensor, Tensor]:
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        x = X_train[samples]
        y = Y_train[samples]
        batch_logits, batch_paths = forward(marketplace, x)
        loss = Tensor.stack(
            *(logits.sparse_categorical_crossentropy(y) for logits in batch_logits),
            dim=0,
        )
        accuracy = Tensor.stack(
            *((logits.sigmoid().argmax(axis=1) == y).sum() for logits in batch_logits),
            dim=0,
        )
        return (
            loss.realize(),
            accuracy.realize(),
            batch_paths.realize(),
        )

    @MultiModelBase.train()
    def combined_forward_step() -> tuple[Tensor, Tensor, Tensor]:
        all_loss = []
        all_accuracy = []
        all_paths = []
        for _ in range(current_forward_pass):
            batch_loss, batch_accuracy, batch_path = forward_step()
            # TODO: we don't need to cat the result, we only need to find the best and keep it
            all_loss.append(batch_loss)
            all_accuracy.append(batch_accuracy)
            all_paths.append(batch_path)
        return (
            Tensor.cat(*all_loss).realize(),
            Tensor.cat(*all_accuracy),
            Tensor.cat(*all_paths).realize(),
        )

    @TinyJit
    def mutate_step(
        combined_loss: Tensor, combined_accuracy: Tensor, combined_paths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        min_loss, min_loss_index = combined_loss.topk(1, largest=False)
        min_path = combined_paths[min_loss_index].flatten()
        min_accuracy = combined_accuracy[min_loss_index]
        mutate(
            marketplace=marketplace,
            leading_path=min_path,
            jitter=lr,
        )
        return min_loss.realize(), min_accuracy.realize(), min_path.realize()

    @TinyJit
    def get_test_acc(path: Tensor) -> Tensor:
        return (
            forward_with_path(marketplace, X_test, path).argmax(axis=1) == Y_test
        ).mean() * 100

    i = 0
    path = None
    test_acc = float("nan")
    current_forward_pass = initial_forward_pass
    for i in (t := trange(step_count)):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing

        if forward_pass_schedule is not None:
            for threshold, forward_pass in reversed(forward_pass_schedule):
                if i >= threshold:
                    if forward_pass != current_forward_pass:
                        combined_forward_step.reset()
                        mutate_step.reset()
                    current_forward_pass = forward_pass
                    break

        start_time = time.perf_counter()

        combined_loss, combined_paths = combined_forward_step()
        loss, accuracy, path = mutate_step(
            combined_loss=combined_loss, combined_paths=combined_paths
        )

        end_time = time.perf_counter()
        run_time = end_time - start_time
        lr.replace(lr * (1 - lr_decay_rate))
        gflops = GlobalCounters.global_ops * 1e-9 / run_time

        if i % metrics_per_steps == (metrics_per_steps - 1):
            test_acc = get_test_acc(path).item()
            mlflow.log_metric("training/loss", loss.item(), step=i)
            mlflow.log_metric("training/accuracy", accuracy.item(), step=i)
            mlflow.log_metric("training/forward_pass", current_forward_pass, step=i)
            mlflow.log_metric("training/lr", lr.item(), step=i)
            mlflow.log_metric("training/gflops", gflops, step=i)
            mlflow.log_metric("testing/accuracy", test_acc, step=i)
        if checkpoint_filepath is not None and i % checkpoint_per_steps == (
            checkpoint_per_steps - 1
        ):
            write_checkpoint(
                marketplace=marketplace,
                path=path,
                global_step=i,
                output_filepath=pathlib.Path(checkpoint_filepath),
            )

        t.set_description(
            f"loss: {loss.item():6.2f}, fw: {current_forward_pass}, rl: {lr.item():e}, "
            f"acc: {test_acc:5.2f}%, {gflops:9,.2f} GFLOPS"
        )
    if path is not None and i is not None and checkpoint_filepath is not None:
        write_checkpoint(
            marketplace=marketplace,
            path=path,
            global_step=i,
            output_filepath=pathlib.Path(checkpoint_filepath),
        )


@click.command("beautiful_mnist")
@click.option("--step-count", type=int, default=10_000, help="How many steps to run")
@click.option("--batch-size", type=int, default=512, help="Size of batch")
@click.option(
    "--initial-lr", type=float, default=1e-3, help="Initial learning rate value"
)
@click.option("--lr-decay", type=float, default=1e-3, help="Learning rate decay rate")
@click.option("--vendor-count", type=int, default=8, help="Vendor count")
@click.option(
    "--checkpoint-filepath",
    type=click.Path(dir_okay=False, writable=True),
    help="Filepath of checkpoint to write to",
)
@click.option(
    "--checkpoint-per-steps",
    type=int,
    default=100,
    help="For how many steps we should write a checkpoint",
)
def main(
    step_count: int,
    batch_size: int,
    initial_lr: float,
    lr_decay: float,
    vendor_count: int,
    checkpoint_filepath: str,
    checkpoint_per_steps: int,
):
    train(
        step_count=step_count,
        batch_size=batch_size,
        initial_lr=initial_lr,
        lr_decay_rate=lr_decay,
        marketplace=make_marketplace(default_vendor_count=vendor_count),
        checkpoint_filepath=pathlib.Path(checkpoint_filepath)
        if checkpoint_filepath is not None
        else None,
        checkpoint_per_steps=checkpoint_per_steps,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exp_id = ensure_experiment("Marketplace")
    with mlflow.start_run(experiment_id=exp_id, run_name="beautiful-mnist"):
        main()
