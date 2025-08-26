# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import functools
import logging
import pathlib
import sys
import time

import click
import mlflow
import numpy as np
from numpy.typing import NDArray
from tinygrad import dtypes
from tinygrad import GlobalCounters
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import getenv
from tinygrad.helpers import trange
from tinygrad.nn import Conv2d
from tinygrad.nn import InstanceNorm
from tinygrad.nn import Linear
from tinygrad.nn.datasets import mnist

from .utils import ensure_experiment
from marketplace.nn import Model
from marketplace.optimizers import Optimizer
from marketplace.training import forward
from marketplace.training import Spec
from marketplace.training import straight_forward
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
            model=Model(
                Conv2d(1, 32, 5),
                Tensor.relu,
                Conv2d(32, 32, 5),
                Tensor.relu,
                InstanceNorm(32),
                Tensor.max_pool2d,
            ),
            vendor_count=structure[0][0],
        ),
        Spec(
            model=Model(
                Conv2d(32, 64, 3),
                Tensor.relu,
                Conv2d(64, 64, 3),
                Tensor.relu,
                InstanceNorm(64),
                Tensor.max_pool2d,
                lambda x: x.flatten(1),
            ),
            vendor_count=structure[1][0],
            upstream_sampling=structure[1][1],
        ),
        Spec(
            model=Model(Linear(576, 10)),
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
    marketplace_replica: int = 1,
    initial_forward_pass: int = 1,
    forward_pass_schedule: list[tuple[int, int]] | None = None,
    meta_lr: float | None = None,
    metrics_per_steps: int = 10,
    checkpoint_filepath: pathlib.Path | None = None,
    checkpoint_per_steps: int = 1000,
    manual_seed: int | None = None,
):
    logger.info(
        "Running beautiful MNIST with step_count=%s, batch_size=%s, init_lr=%s, lr_decay=%s, meta_lr=%s, "
        "marketplace_replica=%s, initial_forward_pass=%s, forward_pass_schedule=%s, metrics_per_steps=%s, "
        "checkpoint_filepath=%s, checkpoint_per_steps=%s, manual_seed=%s",
        step_count,
        batch_size,
        initial_lr,
        lr_decay_rate,
        meta_lr,
        marketplace_replica,
        initial_forward_pass,
        metrics_per_steps,
        forward_pass_schedule,
        checkpoint_filepath,
        checkpoint_per_steps,
        manual_seed,
    )

    mlflow.log_param("step_count", step_count)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("marketplace_replica", marketplace_replica)
    mlflow.log_param("initial_forward_pass", initial_forward_pass)
    mlflow.log_param("lr", initial_lr)
    mlflow.log_param("lr_decay_rate", lr_decay_rate)
    mlflow.log_param("meta_lr", meta_lr)
    mlflow.log_param("forward_pass_schedule", forward_pass_schedule)
    mlflow.log_param("metrics_per_steps", metrics_per_steps)
    mlflow.log_param("checkpoint_per_steps", checkpoint_per_steps)
    mlflow.log_param("manual_seed", manual_seed)

    if manual_seed is not None:
        Tensor.manual_seed(manual_seed)

    X_train, Y_train, X_test, Y_test = load_data()
    lr = Tensor(initial_lr).contiguous().realize()
    optimizer = Optimizer(
        marketplace=marketplace,
        learning_rate=lr,
        meta_learning_rate=Tensor(meta_lr) if meta_lr is not None else None,
    )

    @TinyJit
    def forward_step(samples: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = X_train[samples]
        y = Y_train[samples]
        batch_logits, batch_paths = forward(
            marketplace=marketplace,
            vendors=optimizer.vendors,
            x=x,
        )
        loss = Tensor.stack(
            *(logits.sparse_categorical_crossentropy(y) for logits in batch_logits),
            dim=0,
        )
        accuracy = Tensor.stack(
            *(
                ((logits.sigmoid().argmax(axis=1) == y).sum() / batch_size) * 100
                for logits in batch_logits
            ),
            dim=0,
        )
        return (
            loss.realize(),
            accuracy.realize(),
            batch_paths.realize(),
        )

    def multi_forward_step(sample_batches: Tensor) -> tuple[NDArray, NDArray, NDArray]:
        # TODO: extract this
        product_count = 1
        for spec in reversed(marketplace):
            product_count *= spec.vendor_count
            if spec.upstream_sampling != 0:
                product_count *= spec.upstream_sampling
                break
        # TODO: ideally, if we want to save some memory, we should apply online algorithm here instead so that we don't
        #       need to accumulate all the data in each forward pass
        loss = np.empty([len(sample_batches) * product_count])
        accuracy = np.empty([len(sample_batches) * product_count])
        paths = np.empty(
            [len(sample_batches) * product_count, len(marketplace)], dtype=int
        )

        for i in Tensor.arange(len(sample_batches)):
            i_val = i.item()
            output_slice = slice(i_val * product_count, (i_val + 1) * product_count)
            (
                loss[output_slice],
                accuracy[output_slice],
                paths[output_slice],
            ) = (v.numpy() for v in forward_step(sample_batches[i]))

        unique_paths, indices = np.unique(paths, axis=0, return_inverse=True)
        counts = np.bincount(indices)

        loss_sums = np.bincount(indices, weights=loss)
        loss_means = loss_sums / counts
        accuracy_sums = np.bincount(indices, weights=accuracy)
        accuracy_means = accuracy_sums / counts

        min_idx = loss_means.argmin()
        return (
            loss_means[min_idx],
            accuracy_means[min_idx],
            unique_paths[min_idx],
        )

    @TinyJit
    def optimize_step(seeds: Tensor):
        optimizer.step(seeds)

    @TinyJit
    def get_test_acc() -> Tensor:
        return (
            straight_forward(marketplace, X_test).argmax(axis=1) == Y_test
        ).mean() * 100

    i = 0
    best_seeds = None
    test_acc = float("nan")
    current_forward_pass = initial_forward_pass
    for i in (t := trange(step_count)):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing

        if forward_pass_schedule is not None:
            for threshold, forward_pass in reversed(forward_pass_schedule):
                if i >= threshold:
                    current_forward_pass = forward_pass
                    break

        start_time = time.perf_counter()

        sample_batches = Tensor.randint(
            current_forward_pass, batch_size, high=X_train.shape[0]
        ).realize()

        best_loss, best_accuracy, best_path = multi_forward_step(sample_batches)
        best_seeds = optimizer.get_seeds(Tensor(best_path)).clone().realize()
        for _ in range(marketplace_replica - 1):
            # Update seeds
            Tensor.realize(*optimizer.schedule_seeds_update())
            candidate_loss, candidate_accuracy, candidate_path = multi_forward_step(
                sample_batches
            )
            if candidate_loss.item() >= best_loss.item():
                continue
            best_loss = candidate_loss
            best_accuracy = candidate_accuracy
            best_seeds = optimizer.get_seeds(Tensor(candidate_path)).clone().realize()

        optimize_step(best_seeds)

        end_time = time.perf_counter()
        run_time = end_time - start_time
        lr.assign(lr * (1 - lr_decay_rate)).realize()
        gflops = GlobalCounters.global_ops * 1e-9 / run_time

        if i % metrics_per_steps == (metrics_per_steps - 1):
            test_acc = get_test_acc().item()
            mlflow.log_metric("training/loss", best_loss.item(), step=i)
            mlflow.log_metric("training/accuracy", best_accuracy.item(), step=i)
            mlflow.log_metric("training/forward_pass", current_forward_pass, step=i)
            mlflow.log_metric("training/lr", lr.item(), step=i)
            mlflow.log_metric("training/gflops", gflops, step=i)
            mlflow.log_metric("testing/accuracy", test_acc, step=i)

            if meta_lr is not None:
                mlflow.log_metric("testing/meta_lr", meta_lr, step=i)
                for j, ctx in enumerate(optimizer.spec_context):
                    mlflow.log_metric(
                        f"training/adaptive_lr_{j}", ctx.learning_rate.item(), step=i
                    )

        # if checkpoint_filepath is not None and i % checkpoint_per_steps == (
        #     checkpoint_per_steps - 1
        # ):
        #     write_checkpoint(
        #         marketplace=marketplace,
        #         path=path,
        #         global_step=i,
        #         output_filepath=pathlib.Path(checkpoint_filepath),
        #     )

        t.set_description(
            f"loss: {best_loss.item():6.2f}, fw: {current_forward_pass}, rl: {lr.item():.2e}, "
            f"acc: {best_accuracy.item():.2f}%, vacc: {test_acc:.2f}%, {gflops:9,.2f} GFLOPS"
        )
    # if path is not None and i is not None and checkpoint_filepath is not None:
    #     write_checkpoint(
    #         marketplace=marketplace,
    #         path=path,
    #         global_step=i,
    #         output_filepath=pathlib.Path(checkpoint_filepath),
    #     )


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
@click.option(
    "--marketplace-replica",
    type=int,
    default=1,
    help="How many marketplace replica to run (simulate distributed computing)",
)
@click.option("--vendor-count", type=int, default=8, help="Vendor count")
@click.option("--seed", type=int, help="Set the random seed")
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
    marketplace_replica: int,
    vendor_count: int,
    seed: int | None,
    checkpoint_filepath: str,
    checkpoint_per_steps: int,
):
    # ref: https://github.com/tinygrad/tinygrad/issues/8617
    # With complex huge compute graph, tinygrad runs into recursion too deep issue, let's bump it up
    NEW_RECURSION_LIMIT = 100_000
    logger.info("Current recursion limit is %s", sys.getrecursionlimit())
    sys.setrecursionlimit(NEW_RECURSION_LIMIT)
    logger.info("Set recursion limit to %s", NEW_RECURSION_LIMIT)
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
            manual_seed=seed,
            marketplace=make_marketplace(
                default_vendor_count=vendor_count,
            ),
            marketplace_replica=marketplace_replica,
            checkpoint_filepath=pathlib.Path(checkpoint_filepath)
            if checkpoint_filepath is not None
            else None,
            checkpoint_per_steps=checkpoint_per_steps,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
