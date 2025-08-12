# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import functools
import itertools
import logging
import pathlib
import time

import click
import mlflow
from tinygrad import GlobalCounters
from tinygrad import nn
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import getenv
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import safe_save

from marketplace.multi_nn import MultiConv2d
from marketplace.multi_nn import MultiLinear
from marketplace.multi_nn import MultiModel
from marketplace.multi_nn import SingletonModel
from marketplace.training import forward
from marketplace.training import forward_with_path
from marketplace.training import mutate
from marketplace.training import Spec

logger = logging.getLogger(__name__)


@functools.cache
def load_data():
    return mnist(fashion=getenv("FASHION"))


def make_marketplace(structure: list[tuple[int, int]] | None = None):
    if structure is None:
        structure = [
            # layer 0
            (32, 0),
            # layer 1
            (32, 16),
            # layer 2 (N/A)
            (0, 0),
            # layer 3
            (32, 16),
            # layer 4
            (32, 16),
            # layer 5 (N/A)
            (0, 0),
            # layer 6
            (32, 16),
        ]
    return [
        # layer0
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(structure[0][0], 1, 32, 5),
                    Tensor.relu,
                ]
            ),
        ),
        # layer1
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(structure[1][0], 32, 32, 5),
                    Tensor.relu,
                    Tensor.max_pool2d,
                ]
            ),
            upstream_sampling=structure[1][1],
        ),
        # layer2
        # Spec(
        #     model=MultiModel(
        #         [
        #             # nn.BatchNorm(32),
        #
        #         ],
        #     ),
        #     evolve=False,
        # ),
        # layer3
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(structure[3][0], 32, 64, 3),
                    Tensor.relu,
                ]
            ),
            upstream_sampling=structure[3][1],
        ),
        # layer4
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(structure[4][0], 64, 64, 3),
                    Tensor.relu,
                    Tensor.max_pool2d,
                ]
            ),
            upstream_sampling=structure[4][1],
        ),
        # # layer5
        # Spec(
        #     model=MultiModel(
        #         [
        #             # nn.BatchNorm(64),
        #
        #         ]
        #     ),
        #     evolve=False,
        # ),
        # layer6
        Spec(
            model=MultiModel(
                [lambda x: x.flatten(1), MultiLinear(structure[6][0], 576, 10)]
            ),
            upstream_sampling=structure[6][1],
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
                    SingletonModel(nn.BatchNorm(32)),
                    Tensor.max_pool2d,
                    MultiConv2d(vendor_count, 32, 64, 3),
                    Tensor.relu,
                    MultiConv2d(vendor_count, 64, 64, 3),
                    Tensor.relu,
                    SingletonModel(nn.BatchNorm(64)),
                    Tensor.max_pool2d,
                    lambda x: x.flatten(1),
                    MultiLinear(vendor_count, 576, 10),
                ]
            ),
            excluded_param_keys=frozenset(
                [
                    "layers.4.singleton.weight",
                    "layers.4.singleton.bias",
                    "layers.4.singleton.num_batches_tracked",
                    "layers.4.singleton.running_mean",
                    "layers.4.singleton.running_var",
                    "layers.10.singleton.weight",
                    "layers.10.singleton.bias",
                    "layers.10.singleton.num_batches_tracked",
                    "layers.10.singleton.running_mean",
                    "layers.10.singleton.running_var",
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
    checkpoint_filepath: pathlib.Path | None = None,
    checkpoint_per_steps: int = 1000,
):
    logger.info(
        "Running beautiful MNIST with step_count=%s, batch_size=%s, init_lr=%s, lr_decay=%s, "
        "initial_forward_pass=%s, metrics_per_steps=%s, checkpoint_filepath=%s, checkpoint_per_steps=%s",
        step_count,
        batch_size,
        initial_lr,
        lr_decay_rate,
        initial_forward_pass,
        metrics_per_steps,
        checkpoint_filepath,
        checkpoint_per_steps,
    )

    lr = Tensor(initial_lr)

    mlflow.log_param("step_count", step_count)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("initial_forward_pass", initial_forward_pass)
    mlflow.log_param("lr", initial_lr)
    mlflow.log_param("lr_decay_rate", lr_decay_rate)
    mlflow.log_param("metrics_per_steps", metrics_per_steps)
    mlflow.log_param("checkpoint_per_steps", checkpoint_per_steps)

    X_train, Y_train, X_test, Y_test = load_data()

    @TinyJit
    def forward_step() -> tuple[Tensor, Tensor]:
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        x = X_train[samples]
        y = Y_train[samples]
        batch_logits, batch_paths = forward(marketplace, x)
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
            marketplace=marketplace,
            leading_path=min_path,
            jitter=lr,
        )
        return min_loss.realize(), min_path.realize()

    @TinyJit
    def get_test_acc(path: Tensor) -> Tensor:
        return (
            forward_with_path(marketplace, X_test, path).argmax(axis=1) == Y_test
        ).mean() * 100

    test_acc = float("nan")
    current_forward_pass = initial_forward_pass
    for i in (t := trange(step_count)):
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

        end_time = time.perf_counter()
        run_time = end_time - start_time
        lr.replace(lr * (1 - lr_decay_rate))
        gflops = GlobalCounters.global_ops * 1e-9 / run_time

        if i % metrics_per_steps == (metrics_per_steps - 1):
            test_acc = get_test_acc(path).item()
            mlflow.log_metric("training/loss", loss.item(), step=i)
            mlflow.log_metric("training/accuracy", test_acc, step=i)
            mlflow.log_metric("training/forward_pass", current_forward_pass, step=i)
            mlflow.log_metric("training/lr", lr.item(), step=i)
            mlflow.log_metric("training/gflops", gflops, step=i)
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


def write_checkpoint(
    marketplace: list[Spec],
    path: Tensor,
    global_step: int,
    output_filepath: pathlib.Path,
):
    logger.info(
        "Writing checkpoint with global_step %s to %s", global_step, output_filepath
    )
    parameters = dict(
        itertools.chain.from_iterable(
            [
                (f"layer.{i}.{key}", weights[index])
                for key, weights in get_state_dict(spec.singleton).items()
            ]
            for i, (index, spec) in enumerate(zip(path, marketplace))
        )
    )
    checkpoint_tmp_filepath = output_filepath.with_suffix(".tmp")
    safe_save(
        parameters | dict(global_step=Tensor(global_step)), str(checkpoint_tmp_filepath)
    )
    checkpoint_tmp_filepath.rename(output_filepath)
    logger.info(
        "Wrote checkpoint with global_step %s to %s", global_step, output_filepath
    )


@click.command("beautiful_mnist")
@click.option("--step-count", type=int, default=10_000, help="How many steps to run")
@click.option("--batch-size", type=int, default=32, help="Size of batch")
@click.option(
    "--initial-lr", type=float, default=1e-3, help="Initial learning rate value"
)
@click.option("--lr-decay", type=float, default=1e-3, help="Learning rate decay rate")
@click.option("-c", "--comment", type=str, help="Comment for Tensorboard logs")
def main(
    step_count: int,
    batch_size: int,
    initial_lr: float,
    lr_decay: float,
    comment: str | None,
):
    train(
        step_count=step_count,
        batch_size=batch_size,
        initial_lr=initial_lr,
        lr_decay_rate=lr_decay,
        marketplace=make_marketplace(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with mlflow.start_run():
        main()
