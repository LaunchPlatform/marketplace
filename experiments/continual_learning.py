# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import json
import logging
import pathlib
import sys
import time
import typing
from contextlib import nullcontext

import click
import mlflow
import numpy as np
from PIL import Image
from tinygrad import dtypes
from tinygrad import GlobalCounters
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import trange
from tinygrad.nn import Conv2d
from tinygrad.nn import InstanceNorm
from tinygrad.nn import Linear
from tinygrad.nn.datasets import mnist

from .utils import ensure_experiment
from marketplace.continual_learning import forward_with_paths
from marketplace.nn import Model
from marketplace.optimizers import Optimizer
from marketplace.optimizers import UnitVectorMode
from marketplace.training import Spec
from marketplace.training import straight_forward
from marketplace.utils import load_checkpoint
from marketplace.utils import write_checkpoint

logger = logging.getLogger(__name__)

LABEL_COUNT = 10


# stolen from tinygrad
# ref: https://github.com/tinygrad/tinygrad/blob/c6c16b294616447238d5d19974bceca52c9f2a40/extra/augment.py#L11-L21
def augment_img(
    X: np.typing.NDArray, rotate: float = 10, px: int = 3
) -> np.typing.NDArray:
    Xaug = np.zeros_like(X)
    for i in range(len(X)):
        im = Image.fromarray(X[i])
        im = im.rotate(np.random.randint(-rotate, rotate), resample=Image.BICUBIC)
        w, h = X.shape[1:]
        # upper left, lower left, lower right, upper right
        quad = np.random.randint(-px, px, size=(8)) + np.array([0, 0, 0, h, w, h, w, 0])
        im = im.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)
        Xaug[i] = im
    return Xaug


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
            model=Model(Linear(576, LABEL_COUNT)),
            vendor_count=structure[2][0],
            upstream_sampling=structure[2][1],
        ),
    ]


def learn(
    step_count: int,
    batch_size: int,
    initial_lr: float,
    lr_decay_rate: float,
    marketplace: list[Spec],
    target_new_classes: tuple[int] = (9,),
    probe_scale: float | None = None,
    forward_pass: int = 1,
    metrics_per_steps: int = 10,
    input_checkpoint_filepath: pathlib.Path | None = None,
    checkpoint_filepath: pathlib.Path | None = None,
    checkpoint_per_steps: int = 1000,
    replay_file: typing.TextIO | None = None,
    manual_seed: int | None = None,
):
    logger.info(
        "Running beautiful MNIST continual learning with step_count=%s, batch_size=%s, init_lr=%s, lr_decay=%s, "
        "target_new_classes=%s, probe_scale=%s, forward_pass=%s, metrics_per_steps=%s, input_checkpoint_filepath=%s, "
        "checkpoint_filepath=%s, checkpoint_per_steps=%s, manual_seed=%s",
        step_count,
        batch_size,
        initial_lr,
        lr_decay_rate,
        target_new_classes,
        probe_scale,
        forward_pass,
        metrics_per_steps,
        input_checkpoint_filepath,
        checkpoint_filepath,
        checkpoint_per_steps,
        manual_seed,
    )

    mlflow.log_param("step_count", step_count)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("forward_pass", forward_pass)
    mlflow.log_param("lr", initial_lr)
    mlflow.log_param("lr_decay_rate", lr_decay_rate)
    mlflow.log_param("target_new_classes", target_new_classes)
    mlflow.log_param("probe_scale", probe_scale)
    mlflow.log_param("metrics_per_steps", metrics_per_steps)
    mlflow.log_param("checkpoint_per_steps", checkpoint_per_steps)
    mlflow.log_param("manual_seed", manual_seed)

    if input_checkpoint_filepath is not None:
        load_checkpoint(
            marketplace=marketplace, input_filepath=input_checkpoint_filepath
        )

    if manual_seed is not None:
        Tensor.manual_seed(manual_seed)

    X_train, Y_train, X_test, Y_test = mnist()

    lr = Tensor(initial_lr).contiguous().realize()
    optimizer = Optimizer(
        marketplace=marketplace,
        learning_rate=lr,
        probe_scale=(Tensor(probe_scale) if probe_scale is not None else None),
    )

    @TinyJit
    def forward_step(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_paths = Tensor.stack(
            *(
                Tensor.randint(
                    batch_size, low=0, high=spec.vendor_count, dtype=dtypes.uint
                )
                for spec in marketplace
            ),
            dim=1,
        )

        logits = forward_with_paths(
            marketplace=marketplace,
            paths=batch_paths,
            x=x,
            deltas=[ctx.delta for ctx in optimizer.spec_context],
        )
        loss = logits.sparse_categorical_crossentropy(y, reduction="none")
        correct = logits.argmax(axis=1) == y
        return (
            loss.realize(),
            correct.realize(),
            batch_paths.realize(),
        )

    @TinyJit
    def optimize_step(loss: Tensor, paths: Tensor):
        direction_vectors = optimizer.compute_direction_vectors(
            loss=loss,
            paths=paths,
            unit_vector_mode=UnitVectorMode.whole,
        )
        Tensor.realize(
            *optimizer.schedule_weight_update(
                direction_delta=direction_vectors,
            )
        )
        Tensor.realize(*optimizer.schedule_seeds_update())
        Tensor.realize(*optimizer.schedule_delta_update())

    @TinyJit
    def get_test_acc() -> tuple[Tensor, Tensor]:
        predictions = straight_forward(marketplace, X_test).argmax(axis=1) == Y_test
        new_labels = Y_test == target_new_classes[0]
        for new_label in target_new_classes[1:]:
            new_labels |= Y_test == new_label
        old_labels = ~new_labels
        return (
            # old labels accuracy
            (((predictions & old_labels).sum() / old_labels.sum()) * 100).realize(),
            # new labels accuracy
            (((predictions & new_labels).sum() / new_labels.sum()) * 100).realize(),
        )

    i = 0
    old_test_acc = float("nan")
    new_test_acc = float("nan")
    for i in (t := trange(step_count)):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing

        start_time = time.perf_counter()

        all_samples = []
        all_correct = []
        all_loss = []
        all_paths = []
        all_old_loss = []
        all_old_accuracy = []
        all_new_loss = []
        all_new_accuracy = []
        for _ in range(forward_pass):
            samples = Tensor.randint(
                batch_size, low=0, high=batch_size, dtype=dtypes.uint
            )
            x = X_train[samples]
            y = Y_train[samples]

            loss, correct, paths = forward_step(x, y)
            all_loss.append(loss)
            all_paths.append(paths)

            y = y.numpy()
            loss = loss.numpy()
            correct = correct.numpy()
            samples = samples.numpy()

            old_mask = ~np.isin(y, target_new_classes)
            old_loss = loss[old_mask]
            old_accuracy = correct[old_mask]
            new_mask = ~old_mask
            new_loss = loss[new_mask]
            new_accuracy = correct[new_mask]

            all_samples.append(samples)
            all_correct.append(correct)
            all_old_loss.append(old_loss)
            all_old_accuracy.append(old_accuracy)
            all_new_loss.append(new_loss)
            all_new_accuracy.append(new_accuracy)

        optimize_step(Tensor.cat(*all_loss), Tensor.cat(*all_paths))

        old_loss = np.concatenate(all_old_loss).mean()
        old_accuracy = np.concatenate(all_old_accuracy).mean() * 100
        new_loss = np.concatenate(all_new_loss).mean()
        new_accuracy = np.concatenate(all_new_accuracy).mean() * 100

        end_time = time.perf_counter()
        run_time = end_time - start_time
        lr.assign(lr * (1 - lr_decay_rate)).realize()
        gflops = GlobalCounters.global_ops * 1e-9 / run_time

        if i % metrics_per_steps == (metrics_per_steps - 1):
            old_test_acc, new_test_acc = get_test_acc()
            old_test_acc = old_test_acc.item()
            new_test_acc = new_test_acc.item()
            mlflow.log_metric("learning/old_loss", old_loss.item(), step=i)
            mlflow.log_metric("learning/old_accuracy", old_accuracy.item(), step=i)
            mlflow.log_metric("learning/new_loss", new_loss.item(), step=i)
            mlflow.log_metric("learning/new_accuracy", new_accuracy.item(), step=i)
            mlflow.log_metric("learning/lr", lr.item(), step=i)
            mlflow.log_metric("learning/gflops", gflops, step=i)
            mlflow.log_metric("testing/old_accuracy", old_test_acc, step=i)
            mlflow.log_metric("testing/new_accuracy", new_test_acc, step=i)
            if replay_file is not None:
                replay_file.write(
                    json.dumps(
                        dict(
                            samples=np.concatenate(all_samples).tolist(),
                            correct=np.concatenate(all_correct).tolist(),
                            global_step=i,
                        )
                    )
                )

        if checkpoint_filepath is not None and i % checkpoint_per_steps == (
            checkpoint_per_steps - 1
        ):
            write_checkpoint(
                marketplace=marketplace,
                global_step=i,
                output_filepath=pathlib.Path(checkpoint_filepath),
            )

        t.set_description(
            f"loss: {old_loss.item():6.2f}/{new_loss.item():6.2f}, rl: {lr.item():.2e}, "
            f"acc: {old_accuracy.item():.2f}%/{new_accuracy.item():.2f}%, "
            f"vacc: {old_test_acc:.2f}%/{new_test_acc:.2f}%, {gflops:9,.2f} GFLOPS"
        )
    if i is not None and checkpoint_filepath is not None:
        write_checkpoint(
            marketplace=marketplace,
            global_step=i,
            output_filepath=pathlib.Path(checkpoint_filepath),
        )


@click.command()
@click.option("--step-count", type=int, default=10_000, help="How many steps to run")
@click.option("--batch-size", type=int, default=256, help="Size of batch")
@click.option(
    "--initial-lr", type=float, default=1e-2, help="Initial learning rate value"
)
@click.option("--lr-decay", type=float, default=0, help="Learning rate decay rate")
@click.option("--vendor-count", type=int, default=4, help="Vendor count")
@click.option(
    "--forward-pass",
    type=int,
    default=1,
    help="How many forward pass to run (simulate distributed computing)",
)
@click.option("--seed", type=int, help="Set the random seed")
@click.option(
    "--probe-scale",
    type=float,
    default=1,
    help="The scale we use to apply on LR for making the reconciled delta direction",
)
@click.option(
    "--input-checkpoint-filepath",
    type=click.Path(dir_okay=False, readable=True, exists=True),
    default="continual-learning-v3-exclude-9-neutral.safetensors",
    help="Filepath of checkpoint to read from",
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
@click.option(
    "--replay-filepath",
    type=click.Path(dir_okay=False, writable=True),
    help="Filepath of replay JSON file to write to",
)
@click.option("--run-name", type=str, help="Set the run name")
def main(
    step_count: int,
    batch_size: int,
    initial_lr: float,
    lr_decay: float,
    vendor_count: int,
    forward_pass: int,
    seed: int | None,
    probe_scale: float | None,
    input_checkpoint_filepath: str,
    checkpoint_filepath: str,
    checkpoint_per_steps: int,
    replay_filepath: str | None,
    run_name: str | None,
):
    # ref: https://github.com/tinygrad/tinygrad/issues/8617
    # With complex huge compute graph, tinygrad runs into recursion too deep issue, let's bump it up
    NEW_RECURSION_LIMIT = 100_000
    logger.info("Current recursion limit is %s", sys.getrecursionlimit())
    sys.setrecursionlimit(NEW_RECURSION_LIMIT)
    logger.info("Set recursion limit to %s", NEW_RECURSION_LIMIT)
    if replay_filepath is not None:
        replay_filepath = pathlib.Path(replay_filepath)
        replay_file_ctx = replay_filepath.open("wt")
    else:
        replay_file_ctx = nullcontext()
    with (
        mlflow.start_run(
            experiment_id=ensure_experiment("Continual Learning"),
            run_name="beautiful-mnist" if run_name is None else run_name,
        ),
        replay_file_ctx as replay_file,
    ):
        mlflow.log_param("vendor_count", vendor_count)
        learn(
            step_count=step_count,
            batch_size=batch_size,
            initial_lr=initial_lr,
            lr_decay_rate=lr_decay,
            probe_scale=probe_scale if probe_scale else None,
            forward_pass=forward_pass,
            manual_seed=seed,
            marketplace=make_marketplace(
                default_vendor_count=vendor_count,
            ),
            input_checkpoint_filepath=(
                pathlib.Path(input_checkpoint_filepath)
                if input_checkpoint_filepath is not None
                else None
            ),
            checkpoint_filepath=(
                pathlib.Path(checkpoint_filepath)
                if checkpoint_filepath is not None
                else None
            ),
            checkpoint_per_steps=checkpoint_per_steps,
            replay_file=replay_file,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
