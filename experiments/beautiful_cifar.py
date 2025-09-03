# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import functools
import logging
import pathlib
import sys
import time

import click
import mlflow
from tinygrad import dtypes
from tinygrad import GlobalCounters
from tinygrad import nn
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import getenv
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist

from .utils import ensure_experiment
from marketplace.nn import Model
from marketplace.optimizers import Optimizer
from marketplace.training import forward
from marketplace.training import Spec
from marketplace.training import straight_forward
from marketplace.utils import write_checkpoint

logger = logging.getLogger(__name__)

batchsize = getenv("BS", 1024)
bias_scaler = 64
hyp = {
    "opt": {
        "bias_lr": 1.525
        * bias_scaler
        / 512,  # TODO: Is there maybe a better way to express the bias and batchnorm scaling? :'))))
        "non_bias_lr": 1.525 / 512,
        "bias_decay": 6.687e-4 * batchsize / bias_scaler,
        "non_bias_decay": 6.687e-4 * batchsize,
        "scaling_factor": 1.0 / 9,
        "percent_start": 0.23,
        "loss_scale_scaler": 1.0
        / 32,  # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
    },
    "net": {
        "whitening": {
            "kernel_size": 2,
            "num_examples": 50000,
        },
        "batch_norm_momentum": 0.4,  # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        "cutmix_size": 3,
        "cutmix_epochs": 6,
        "pad_amount": 2,
        "base_depth": 64,  ## This should be a factor of 8 in some way to stay tensor core friendly
    },
    "misc": {
        "ema": {
            "epochs": 10,  # Slight bug in that this counts only full epochs and then additionally runs the EMA for any fractional epochs at the end too
            "decay_base": 0.95,
            "decay_pow": 3.0,
            "every_n_steps": 5,
        },
        "train_epochs": 12,
        #'train_epochs': 12.1,
        "device": "cuda",
        "data_location": "data.pt",
    },
}
whiten_conv_depth = 3 * hyp["net"]["whitening"]["kernel_size"] ** 2

scaler = 2.0  ## You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
depths = {
    "init": round(scaler**-1 * hyp["net"]["base_depth"]),  # 32  w/ scaler at base value
    "block1": round(
        scaler**0 * hyp["net"]["base_depth"]
    ),  # 64  w/ scaler at base value
    "block2": round(
        scaler**2 * hyp["net"]["base_depth"]
    ),  # 256 w/ scaler at base value
    "block3": round(
        scaler**3 * hyp["net"]["base_depth"]
    ),  # 512 w/ scaler at base value
    "num_classes": 10,
}


class ConvGroup:
    def __init__(self, channels_in, channels_out):
        self.conv1 = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            channels_out, channels_out, kernel_size=3, padding=1, bias=False
        )

        # self.norm1 = nn.BatchNorm(channels_out, track_running_stats=False, eps=1e-12, momentum=hyp['net']['batch_norm_momentum'])
        # self.norm2 = nn.BatchNorm(channels_out, track_running_stats=False, eps=1e-12, momentum=hyp['net']['batch_norm_momentum'])
        # Notice: Swap to use instance norm because batch norm doesn't work well with marketplace
        self.norm1 = nn.InstanceNorm(channels_out)
        self.norm2 = nn.InstanceNorm(channels_out)

    def __call__(self, x: Tensor) -> Tensor:
        x = (
            self.norm1(self.conv1(x).max_pool2d().float())
            .cast(dtypes.default_float)
            .quick_gelu()
        )
        return (
            self.norm2(self.conv2(x).float()).cast(dtypes.default_float).quick_gelu()
            + x
        )


X_train, Y_train, X_test, Y_test = nn.datasets.cifar()
cifar10_std, cifar10_mean = X_train.float().std_mean(axis=(0, 2, 3))


def preprocess(X: Tensor) -> Tensor:
    return ((X - cifar10_mean.view(1, -1, 1, 1)) / cifar10_std.view(1, -1, 1, 1)).cast(
        dtypes.default_float
    )


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
                # whiten
                # ref: https://github.com/tinygrad/tinygrad/blob/8a2846b31a47cae96f49f5e4671f54aa58293e00/examples/beautiful_cifar.py#L84C23-L84C36
                nn.Conv2d(
                    3,
                    2 * whiten_conv_depth,
                    kernel_size=hyp["net"]["whitening"]["kernel_size"],
                    padding=0,
                    bias=False,
                ),
                Tensor.quick_gelu,
                # ref: https://github.com/tinygrad/tinygrad/blob/8a2846b31a47cae96f49f5e4671f54aa58293e00/examples/beautiful_cifar.py#L86
                lambda x: x.pad((1, 0, 0, 1)),
                # conv_group_1
                # ref: https://github.com/tinygrad/tinygrad/blob/8a2846b31a47cae96f49f5e4671f54aa58293e00/examples/beautiful_cifar.py#L79C10-L79C23
                ConvGroup(2 * whiten_conv_depth, depths["block1"]),
            ),
            vendor_count=structure[0][0],
        ),
        Spec(
            model=Model(
                ConvGroup(depths["block1"], depths["block2"]),
            ),
            vendor_count=structure[1][0],
            upstream_sampling=structure[1][1],
        ),
        Spec(
            model=Model(
                ConvGroup(depths["block2"], depths["block3"]),
                lambda x: x.max(axis=(2, 3)),
                nn.Linear(depths["block3"], depths["num_classes"], bias=False),
                # ref: https://github.com/tinygrad/tinygrad/blob/8a2846b31a47cae96f49f5e4671f54aa58293e00/examples/beautiful_cifar.py#L89
                lambda x: x * hyp["opt"]["scaling_factor"],
            ),
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
    meta_lr: float | None = None,
    probe_scale: float | None = None,
    marketplace_replica: int = 1,
    initial_forward_pass: int = 1,
    forward_pass_schedule: list[tuple[int, int]] | None = None,
    metrics_per_steps: int = 10,
    checkpoint_filepath: pathlib.Path | None = None,
    checkpoint_per_steps: int = 1000,
    manual_seed: int | None = None,
):
    logger.info(
        "Running beautiful CIFAR with step_count=%s, batch_size=%s, init_lr=%s, lr_decay=%s, meta_lr=%s, "
        "probe_scale=%s, marketplace_replica=%s, initial_forward_pass=%s, forward_pass_schedule=%s, "
        "metrics_per_steps=%s, checkpoint_filepath=%s, checkpoint_per_steps=%s, manual_seed=%s",
        step_count,
        batch_size,
        initial_lr,
        lr_decay_rate,
        meta_lr,
        probe_scale,
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
    mlflow.log_param("probe_scale", probe_scale)
    mlflow.log_param("forward_pass_schedule", forward_pass_schedule)
    mlflow.log_param("metrics_per_steps", metrics_per_steps)
    mlflow.log_param("checkpoint_per_steps", checkpoint_per_steps)
    mlflow.log_param("manual_seed", manual_seed)

    if manual_seed is not None:
        Tensor.manual_seed(manual_seed)

    lr = Tensor(initial_lr).contiguous().realize()
    optimizer = Optimizer(
        marketplace=marketplace,
        learning_rate=lr,
        probe_scale=(Tensor(probe_scale) if probe_scale is not None else None),
        meta_learning_rate=(Tensor(meta_lr) if meta_lr is not None else None),
    )

    @TinyJit
    def forward_step(samples: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = X_train[samples]
        y = Y_train[samples]
        batch_logits, batch_paths = forward(
            marketplace=marketplace,
            vendors=optimizer.vendors,
            x=preprocess(x),
        )
        loss = Tensor.stack(
            *(logits.sparse_categorical_crossentropy(y) for logits in batch_logits),
            dim=0,
        )
        accuracy = Tensor.stack(
            *(
                ((logits.argmax(axis=1) == y).sum() / batch_size) * 100
                for logits in batch_logits
            ),
            dim=0,
        )
        return (
            loss.realize(),
            accuracy.realize(),
            batch_paths.realize(),
        )

    @TinyJit
    def compute_direction_vectors(
        loss: Tensor, paths: Tensor
    ) -> list[dict[str, Tensor]]:
        direction_vectors = optimizer.compute_direction_vectors(
            loss=loss,
            paths=paths,
        )
        # TODO: optional
        Tensor.realize(*optimizer.schedule_lr_scale_update(direction_vectors))
        return [
            {key: params.realize() for key, params in delta.items()}
            for delta in direction_vectors
        ]

    @TinyJit
    def lr_scale_optimize_step(
        direction_vectors: list[dict[str, Tensor]] | None, learning_rates: Tensor | None
    ):
        Tensor.realize(
            *optimizer.schedule_weight_update(
                direction_delta=direction_vectors, learning_rates=learning_rates
            )
        )
        Tensor.realize(*optimizer.schedule_seeds_update())
        Tensor.realize(*optimizer.schedule_delta_update())

    @TinyJit
    def optimize_step(
        samples: Tensor, loss: Tensor, paths: Tensor
    ) -> tuple[Tensor, Tensor]:
        direction_vectors = optimizer.compute_direction_vectors(
            loss=loss,
            paths=paths,
        )
        Tensor.realize(
            *optimizer.schedule_weight_update(
                direction_delta=direction_vectors,
            )
        )
        Tensor.realize(*optimizer.schedule_seeds_update())
        Tensor.realize(*optimizer.schedule_delta_update())

        # let's run forward pass again to see accuracy and loss
        x = X_train[samples]
        y = Y_train[samples]
        logits = straight_forward(marketplace, preprocess(x))
        loss = logits.sparse_categorical_crossentropy(y)
        accuracy = ((logits.argmax(axis=1) == y).sum() / batch_size) * 100
        return loss.realize(), accuracy.realize()

    @TinyJit
    def get_test_acc() -> Tensor:
        return (
            straight_forward(marketplace, preprocess(X_test)).argmax(axis=1) == Y_test
        ).mean() * 100

    i = 0
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

        # direction probing forward pass
        loss, _, paths = forward_step(sample_batches[0])

        if meta_lr is not None:
            # lr scaling forward pass
            direction_vectors = compute_direction_vectors(loss=loss, paths=paths)
            loss, accuracy, paths = forward_step(sample_batches[0])
            best_loss, best_index = loss.topk(1, largest=False)
            best_index = best_index.squeeze(0)
            best_accuracy = accuracy[best_index]
            best_path = paths[best_index]
            best_lr = optimizer.get_learning_rates(best_path)
            lr_scale_optimize_step(direction_vectors, best_lr)
        else:
            best_loss, best_accuracy = optimize_step(
                samples=sample_batches[0], loss=loss, paths=paths
            )

        end_time = time.perf_counter()
        run_time = end_time - start_time
        if meta_lr is not None:
            optimizer.meta_learning_rate.assign(
                optimizer.meta_learning_rate * (1 - lr_decay_rate)
            ).realize()
        else:
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
                mlflow.log_metric(
                    "testing/meta_lr", optimizer.meta_learning_rate.item(), step=i
                )
                for j, spec_lr in enumerate(best_lr):
                    mlflow.log_metric(
                        f"training/adaptive_lr_{j}", spec_lr.item(), step=i
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
            f"loss: {best_loss.item():6.2f}, fw: {current_forward_pass}, rl: {lr.item():.2e}, "
            f"acc: {best_accuracy.item():.2f}%, vacc: {test_acc:.2f}%, {gflops:9,.2f} GFLOPS"
        )
    if i is not None and checkpoint_filepath is not None:
        write_checkpoint(
            marketplace=marketplace,
            global_step=i,
            output_filepath=pathlib.Path(checkpoint_filepath),
        )


@click.command("beautiful_mnist")
@click.option("--step-count", type=int, default=10_000, help="How many steps to run")
@click.option("--batch-size", type=int, default=512, help="Size of batch")
@click.option(
    "--initial-lr", type=float, default=1e-3, help="Initial learning rate value"
)
@click.option("--lr-decay", type=float, default=1e-5, help="Learning rate decay rate")
@click.option(
    "--meta-lr",
    type=click.FloatRange(0.0, 1.0, max_open=True),
    help="Enable learning rate scaling mode with the given meta-learning rate",
)
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
@click.option("--vendor-count", type=int, default=4, help="Vendor count")
@click.option("--seed", type=int, help="Set the random seed")
@click.option(
    "--probe-scale",
    type=float,
    default=0.1,
    help="The scale we use to apply on LR for making the reconciled delta direction",
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
@click.option("--run-name", type=str, help="Set the run name")
def main(
    step_count: int,
    batch_size: int,
    initial_lr: float,
    lr_decay: float,
    meta_lr: float | None,
    forward_pass: int,
    marketplace_replica: int,
    vendor_count: int,
    seed: int | None,
    probe_scale: float | None,
    checkpoint_filepath: str,
    checkpoint_per_steps: int,
    run_name: str | None,
):
    # ref: https://github.com/tinygrad/tinygrad/issues/8617
    # With complex huge compute graph, tinygrad runs into recursion too deep issue, let's bump it up
    NEW_RECURSION_LIMIT = 100_000
    logger.info("Current recursion limit is %s", sys.getrecursionlimit())
    sys.setrecursionlimit(NEW_RECURSION_LIMIT)
    logger.info("Set recursion limit to %s", NEW_RECURSION_LIMIT)
    with mlflow.start_run(
        experiment_id=ensure_experiment("Marketplace CIFAR"),
        run_name="beautiful-cifar" if run_name is None else run_name,
    ):
        mlflow.log_param("vendor_count", vendor_count)
        train(
            step_count=step_count,
            batch_size=batch_size,
            initial_lr=initial_lr,
            lr_decay_rate=lr_decay,
            meta_lr=meta_lr,
            initial_forward_pass=forward_pass,
            probe_scale=probe_scale if probe_scale else None,
            manual_seed=seed,
            marketplace=make_marketplace(
                default_vendor_count=vendor_count,
            ),
            marketplace_replica=marketplace_replica,
            checkpoint_filepath=(
                pathlib.Path(checkpoint_filepath)
                if checkpoint_filepath is not None
                else None
            ),
            checkpoint_per_steps=checkpoint_per_steps,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
