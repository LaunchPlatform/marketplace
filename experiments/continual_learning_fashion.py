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
    target_new_classes: tuple[int] = (3,),
    balance_labels: bool = True,
    augment_old: bool = True,
    augment_new: bool = True,
    new_train_size: int = 8,
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
        "target_new_classes=%s, balance_labels=%s, augment_old=%s, augment_new=%s, new_train_size=%s, probe_scale=%s, "
        "forward_pass=%s, metrics_per_steps=%s, input_checkpoint_filepath=%s, checkpoint_filepath=%s, "
        "checkpoint_per_steps=%s, manual_seed=%s",
        step_count,
        batch_size,
        initial_lr,
        lr_decay_rate,
        target_new_classes,
        balance_labels,
        augment_old,
        augment_new,
        new_train_size,
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
    mlflow.log_param("balance_labels", balance_labels)
    mlflow.log_param("augment_old", augment_old)
    mlflow.log_param("augment_new", augment_new)
    mlflow.log_param("new_train_size", new_train_size)
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
    new_X_train, new_Y_train, new_X_test, new_Y_test = mnist(fashion=True)

    if target_new_classes is not None:
        class_mask = np.isin(new_Y_train.numpy(), target_new_classes)
        target_new_X_train = Tensor(new_X_train.numpy()[class_mask])
        target_new_Y_train = Tensor(new_Y_train.numpy()[class_mask])
        class_mask = np.isin(new_Y_test.numpy(), target_new_classes)
        target_new_X_test = Tensor(new_X_test.numpy()[class_mask])
        target_new_Y_test = Tensor(new_Y_test.numpy()[class_mask])
    else:
        target_new_X_train = new_X_train
        target_new_Y_train = new_Y_train
        target_new_X_test = new_X_test
        target_new_Y_test = new_Y_test

    lr = Tensor(initial_lr).contiguous().realize()
    optimizer = Optimizer(
        marketplace=marketplace,
        learning_rate=lr,
        probe_scale=(Tensor(probe_scale) if probe_scale is not None else None),
    )

    @TinyJit
    def forward_step(
        old_x: Tensor,
        old_y: Tensor,
        new_x: Tensor,
        new_y: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        combined_x = Tensor.cat(old_x, new_x)
        combined_y = Tensor.cat(old_y, new_y)

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
            x=combined_x,
            deltas=[ctx.delta for ctx in optimizer.spec_context],
        )
        loss = logits.sparse_categorical_crossentropy(combined_y, reduction="none")

        if balance_labels:
            # Notice: by adding the target classes from new dataset, we are changing the probability of each number
            # appearing. Not sure if it matters, but to make the model harder to blindly guess, we are balancing the
            # unbalanced labels by introducing the cross entropy weights.
            weights = np.repeat((1 / LABEL_COUNT) * len(old_x), LABEL_COUNT)
            weights[target_new_classes] += (1 / len(target_new_classes)) * len(new_x)
            weights = batch_size / weights
            max_weight = weights.max()
            weights = Tensor(weights / max_weight, dtype=dtypes.default_float)
            loss *= weights[combined_y]
        correct = logits.argmax(axis=1) == combined_y
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
        old = (
            straight_forward(marketplace, X_test).argmax(axis=1) == Y_test
        ).mean() * 100
        new = (
            straight_forward(marketplace, target_new_X_test).argmax(axis=1)
            == target_new_Y_test
        ).mean() * 100
        return old.realize(), new.realize()

    i = 0
    old_test_acc = float("nan")
    new_test_acc = float("nan")
    for i in (t := trange(step_count)):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing

        start_time = time.perf_counter()
        old_train_size = batch_size - new_train_size

        all_old_samples = []
        all_new_samples = []
        all_old_correct = []
        all_new_correct = []
        all_loss = []
        all_paths = []
        all_old_loss = []
        all_old_accuracy = []
        all_new_loss = []
        all_new_accuracy = []
        for _ in range(forward_pass):
            old_samples = Tensor.randint(
                old_train_size, low=0, high=old_train_size, dtype=dtypes.uint
            )
            new_samples = Tensor.randint(
                new_train_size, low=0, high=new_train_size, dtype=dtypes.uint
            )
            old_x = X_train[old_samples]
            old_y = Y_train[old_samples]
            new_x = target_new_X_train[new_samples]
            new_y = target_new_Y_train[new_samples]
            # TODO: a bit slow, ideally run with a background loader
            if augment_old:
                old_x = old_x.reshape(-1, 28, 28).numpy().astype(np.uint8)
                old_x = Tensor(
                    augment_img(old_x).reshape(-1, 1, 28, 28),
                    dtype=dtypes.default_float,
                )
            if augment_new:
                new_x = new_x.reshape(-1, 28, 28).numpy().astype(np.uint8)
                new_x = Tensor(
                    augment_img(new_x).reshape(-1, 1, 28, 28),
                    dtype=dtypes.default_float,
                )

            loss, correct, paths = forward_step(
                old_x=old_x,
                old_y=old_y,
                new_x=new_x,
                new_y=new_y,
            )
            old_loss = loss[:old_train_size].mean()
            old_correct = correct[:old_train_size]
            old_accuracy = old_correct.mean()

            new_loss = loss[old_train_size:].mean()
            new_correct = correct[old_train_size:]
            new_accuracy = new_correct.mean()

            all_old_samples.append(old_samples.numpy())
            all_new_samples.append(new_samples.numpy())
            all_loss.append(loss)
            all_paths.append(paths)
            all_old_loss.append(old_loss.numpy())
            all_old_correct.append(old_correct.numpy())
            all_old_accuracy.append(old_accuracy.numpy())
            all_new_loss.append(new_loss.numpy())
            all_new_correct.append(new_correct.numpy())
            all_new_accuracy.append(new_accuracy.numpy())

        optimize_step(Tensor.cat(*all_loss), Tensor.cat(*all_paths))

        old_loss = np.array(all_old_loss).mean()
        old_accuracy = np.array(all_old_accuracy).mean() * 100
        new_loss = np.array(all_new_loss).mean()
        new_accuracy = np.array(all_new_accuracy).mean() * 100

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
                            old_samples=np.concatenate(all_old_samples).tolist(),
                            old_correct=np.concatenate(all_old_correct).tolist(),
                            new_samples=np.concatenate(all_new_samples).tolist(),
                            new_correct=np.concatenate(all_new_correct).tolist(),
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
    default="continual-learning.safetensors",
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
    "--replay-file",
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
    replay_file: str | None,
    run_name: str | None,
):
    # ref: https://github.com/tinygrad/tinygrad/issues/8617
    # With complex huge compute graph, tinygrad runs into recursion too deep issue, let's bump it up
    NEW_RECURSION_LIMIT = 100_000
    logger.info("Current recursion limit is %s", sys.getrecursionlimit())
    sys.setrecursionlimit(NEW_RECURSION_LIMIT)
    logger.info("Set recursion limit to %s", NEW_RECURSION_LIMIT)
    if replay_file is not None:
        replay_filepath = pathlib.Path(replay_file)
        replay_file_ctx = replay_filepath.open("wt")
    else:
        replay_file_ctx = nullcontext()
    with (
        mlflow.start_run(
            experiment_id=ensure_experiment("Continual Learning - Fashion"),
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
