# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import functools
import logging
import pathlib
import sys
import time
import typing

import click
import mlflow
from tinygrad import Device
from tinygrad import GlobalCounters
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import getenv
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import get_state_dict

from .utils import ensure_experiment
from marketplace.delta_nn import DeltaConv2d
from marketplace.delta_nn import DeltaInstanceNorm
from marketplace.delta_nn import DeltaLinear
from marketplace.delta_nn import DeltaModel
from marketplace.delta_nn import DeltaModelBase
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
    return [
        Spec(
            model=DeltaModel(
                [
                    DeltaConv2d(1, 32, 5),
                    Tensor.relu,
                    DeltaConv2d(32, 32, 5),
                    Tensor.relu,
                    DeltaInstanceNorm(32),
                    Tensor.max_pool2d,
                ]
            ),
            vendor_count=structure[0][0],
        ),
        Spec(
            model=DeltaModel(
                [
                    DeltaConv2d(32, 64, 3),
                    Tensor.relu,
                    DeltaConv2d(64, 64, 3),
                    Tensor.relu,
                    DeltaInstanceNorm(64),
                    Tensor.max_pool2d,
                    lambda x: x.flatten(1),
                ]
            ),
            vendor_count=structure[1][0],
            upstream_sampling=structure[1][1],
        ),
        Spec(
            model=DeltaModel([DeltaLinear(576, 10)]),
            vendor_count=structure[2][0],
            upstream_sampling=structure[2][1],
        ),
    ]


def train(
    step_count: int,
    batch_size: int,
    initial_lr: float,
    lr_decay_rate: float,
    marketplace: list[Spec],
    vendor_devices: tuple[str] | None = None,
    initial_forward_pass: int = 1,
    metrics_per_steps: int = 10,
    forward_pass_schedule: list[tuple[int, int]] | None = None,
    checkpoint_filepath: pathlib.Path | None = None,
    checkpoint_per_steps: int = 1000,
):
    logger.info(
        "Running beautiful MNIST with step_count=%s, batch_size=%s, init_lr=%s, lr_decay=%s, "
        "vendor_devices=%s, initial_forward_pass=%s, metrics_per_steps=%s, forward_pass_schedule=%s, "
        "checkpoint_filepath=%s, checkpoint_per_steps=%s",
        step_count,
        batch_size,
        initial_lr,
        lr_decay_rate,
        vendor_devices,
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
    mlflow.log_param("vendor_devices", vendor_devices)
    mlflow.log_param("forward_pass_schedule", forward_pass_schedule)
    mlflow.log_param("metrics_per_steps", metrics_per_steps)
    mlflow.log_param("checkpoint_per_steps", checkpoint_per_steps)

    X_train, Y_train, X_test, Y_test = load_data()

    if vendor_devices:
        for spec in marketplace:
            for key, param in get_state_dict(spec.model).items():
                param.shard_(vendor_devices, axis=0)
        X_train.to_(vendor_devices)
        Y_train.to_(vendor_devices)
        X_test.to_(vendor_devices)
        Y_test.to_(vendor_devices)

    @TinyJit
    @DeltaModelBase.train()
    def forward_step() -> tuple[Tensor, Tensor, Tensor]:
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        x = X_train[samples]
        y = Y_train[samples]
        batch_logits, batch_seeds = forward(marketplace, x)
        loss = Tensor.stack(
            *(logits.sparse_categorical_crossentropy(y) for logits in batch_logits),
            dim=0,
        )
        best_loss, best_index = loss.topk(1, largest=False)
        best_index = best_index.squeeze(0)
        accuracy = (
            (batch_logits[best_index].sigmoid().argmax(axis=1) == y).sum() / batch_size
        ) * 100
        return (
            best_loss.realize(),
            accuracy.realize(),
            batch_seeds[best_index].realize(),
        )

    @TinyJit
    def mutate_step(best_seeds: Tensor):
        mutate(
            marketplace=marketplace,
            best_seeds=best_seeds,
            jitter=lr,
        )

    @TinyJit
    def get_test_acc(path: Tensor) -> Tensor:
        return (
            forward_with_path(marketplace, X_test, path).argmax(axis=1) == Y_test
        ).mean() * 100

    i = 0
    seeds = None
    test_acc = float("nan")
    current_forward_pass = initial_forward_pass
    for i in (t := trange(step_count)):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing

        if forward_pass_schedule is not None:
            for threshold, forward_pass in reversed(forward_pass_schedule):
                if i >= threshold:
                    if forward_pass != current_forward_pass:
                        mutate_step.reset()
                    current_forward_pass = forward_pass
                    break

        start_time = time.perf_counter()

        best_loss, best_accuracy, seeds = forward_step()
        for _ in range(current_forward_pass - 1):
            batch_loss, batch_accuracy, batch_seeds = forward_step()
            if batch_loss.item() >= best_loss.item():
                continue
            best_loss = batch_loss
            best_accuracy = batch_accuracy
            seeds = batch_seeds

        mutate_step(seeds)

        end_time = time.perf_counter()
        run_time = end_time - start_time
        lr.assign(lr * (1 - lr_decay_rate)).realize()
        gflops = GlobalCounters.global_ops * 1e-9 / run_time

        if i % metrics_per_steps == (metrics_per_steps - 1):
            test_acc = get_test_acc(path).item()
            mlflow.log_metric("training/loss", best_loss.item(), step=i)
            mlflow.log_metric("training/accuracy", best_accuracy.item(), step=i)
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
            f"loss: {best_loss.item():6.2f}, fw: {current_forward_pass}, rl: {lr.item():.2e}, "
            f"acc: {best_accuracy.item():.2f}%, vacc: {test_acc:.2f}%, {gflops:9,.2f} GFLOPS"
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
@click.option("--lr-decay", type=float, default=1e-4, help="Learning rate decay rate")
@click.option(
    "--forward-pass",
    type=int,
    default=1,
    help="How many forward pass to run (simulate distributed computing)",
)
@click.option("--vendor-count", type=int, default=8, help="Vendor count")
@click.option(
    "--gpus",
    type=int,
    help="Distribute vendors to multiple GPUs to speed up the training (not working yet)",
)
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
    forward_pass: int,
    vendor_count: int,
    gpus: int,
    checkpoint_filepath: str,
    checkpoint_per_steps: int,
):
    # ref: https://github.com/tinygrad/tinygrad/issues/8617
    # With complex huge compute graph, tinygrad runs into recursion too deep issue, let's bump it up
    NEW_RECURSION_LIMIT = 100_000
    logger.info("Current recursion limit is %s", sys.getrecursionlimit())
    sys.setrecursionlimit(NEW_RECURSION_LIMIT)
    logger.info("Set recursion limit to %s", NEW_RECURSION_LIMIT)
    vendor_devices = None
    if gpus:
        vendor_devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(gpus))
    with mlflow.start_run(
        experiment_id=ensure_experiment("Marketplace V2"),
        run_name="beautiful-mnist",
    ):
        train(
            step_count=step_count,
            batch_size=batch_size,
            initial_lr=initial_lr,
            lr_decay_rate=lr_decay,
            initial_forward_pass=forward_pass,
            marketplace=make_marketplace(
                default_vendor_count=vendor_count,
            ),
            vendor_devices=vendor_devices,
            checkpoint_filepath=pathlib.Path(checkpoint_filepath)
            if checkpoint_filepath is not None
            else None,
            checkpoint_per_steps=checkpoint_per_steps,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
