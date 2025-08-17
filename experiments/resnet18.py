import glob
import json
import pathlib
import random
import time
import typing

import mlflow
import numpy as np
from PIL import Image
from tinygrad import nn
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import GlobalCounters
from tinygrad.helpers import trange
from tinyloader.loader import load
from tinyloader.loader import load_with_workers
from tinyloader.loader import Loader

from .utils import ensure_experiment
from marketplace.multi_nn import MultiConv2d
from marketplace.multi_nn import MultiInstanceNorm
from marketplace.multi_nn import MultiLinear
from marketplace.multi_nn import MultiModel
from marketplace.multi_nn import MultiModelBase
from marketplace.training import forward
from marketplace.training import forward_with_path
from marketplace.training import mutate
from marketplace.training import Spec


def get_train_files(basedir: pathlib.Path) -> list[str]:
    if not (files := glob.glob(p := str(basedir / "train/*/*"))):
        raise FileNotFoundError(f"No training files in {p}")
    return files


def get_val_files(basedir: pathlib.Path) -> list[str]:
    if not (files := glob.glob(p := str(basedir / "val/*/*"))):
        raise FileNotFoundError(f"No training files in {p}")
    return files


def get_imagenet_categories(basedir: pathlib.Path) -> dict[str, int]:
    ci = json.load(open(basedir / "imagenet_class_index.json"))
    return {v[0]: int(k) for k, v in ci.items()}


def center_crop(img: Image) -> Image:
    rescale = min(img.size) / 256
    crop_left = (img.width - 224 * rescale) / 2.0
    crop_top = (img.height - 224 * rescale) / 2.0
    img = img.resize(
        (224, 224),
        Image.BILINEAR,
        box=(crop_left, crop_top, crop_left + 224 * rescale, crop_top + 224 * rescale),
    )
    return img


class ImageLoader(Loader):
    def __init__(self, img_categories: dict[str, int]):
        super().__init__()
        self.img_categories = img_categories

    def make_request(self, item: pathlib.Path) -> typing.Any:
        return item

    def load(self, request: pathlib.Path) -> tuple[np.typing.NDArray, ...]:
        x = Image.open(request)
        x = center_crop(x)
        x = np.transpose(np.asarray(x), (2, 0, 1))
        y = self.img_categories[request.parts[-2]]
        return x, np.array(y)

    def post_process(
        self, response: tuple[np.typing.NDArray, ...]
    ) -> tuple[Tensor, ...]:
        x, y = response
        x = Tensor(x.copy()).contiguous().realize()
        y = Tensor(y).realize()
        return x, y


class BasicBlock(MultiModelBase):
    def __init__(
        self, vendor_count: int, in_channels: int, out_channels: int, stride: int = 1
    ):
        super().__init__()
        self.vendor_count = vendor_count
        self.conv1 = MultiConv2d(
            vendor_count,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = MultiInstanceNorm(vendor_count, out_channels)
        self.conv2 = MultiConv2d(
            vendor_count,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = MultiInstanceNorm(vendor_count, out_channels)

        self.downsample = lambda i, x: x
        if stride != 1 or in_channels != out_channels:
            self.downsample = MultiModel(
                [
                    MultiConv2d(
                        vendor_count,
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    MultiInstanceNorm(vendor_count, out_channels),
                ]
            )

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        out = self.conv1(i, x)
        out = self.bn1(i, out)
        out = out.relu()
        out = self.conv2(i, out)
        out = self.bn2(i, out)
        out += self.downsample(i, x)
        out = out.relu()
        return out


def make_marketplace(num_classes: int = 100, default_vendor_count: int = 4):
    layer0_vendor_count = default_vendor_count
    layer1_upstream_sampling = 0
    layer1_vendor_count = default_vendor_count
    layer2_upstream_sampling = 0
    layer2_vendor_count = default_vendor_count
    return [
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(
                        layer0_vendor_count,
                        in_channels=3,
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=False,
                    ),
                    MultiInstanceNorm(layer0_vendor_count, 64),
                    Tensor.relu,
                    lambda x: x.max_pool2d(
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    BasicBlock(
                        layer0_vendor_count,
                        in_channels=64,
                        out_channels=64,
                        stride=1,
                    ),
                    BasicBlock(
                        layer0_vendor_count,
                        in_channels=64,
                        out_channels=64,
                        stride=1,
                    ),
                ]
            ),
            upstream_sampling=2,
        ),
        # layer1
        Spec(
            model=MultiModel(
                [
                    BasicBlock(
                        layer1_vendor_count,
                        in_channels=64,
                        out_channels=128,
                        stride=2,
                    ),
                    BasicBlock(
                        layer1_vendor_count,
                        in_channels=128,
                        out_channels=128,
                        stride=1,
                    ),
                    BasicBlock(
                        layer1_vendor_count,
                        in_channels=128,
                        out_channels=256,
                        stride=2,
                    ),
                    BasicBlock(
                        layer1_vendor_count,
                        in_channels=256,
                        out_channels=256,
                        stride=1,
                    ),
                ]
            ),
            upstream_sampling=layer1_upstream_sampling,
        ),
        # layer4
        Spec(
            model=MultiModel(
                [
                    BasicBlock(
                        layer2_vendor_count,
                        in_channels=256,
                        out_channels=512,
                        stride=2,
                    ),
                    BasicBlock(
                        layer2_vendor_count,
                        in_channels=512,
                        out_channels=512,
                        stride=1,
                    ),
                    lambda x: x.avg_pool2d(kernel_size=7),
                    lambda x: x.flatten(1),
                    MultiLinear(layer2_vendor_count, 512, num_classes),
                ]
            ),
            upstream_sampling=layer2_upstream_sampling,
        ),
    ]


def train(
    dataset_dir: pathlib.Path,
    marketplace: list[Spec],
    step_count: int = 100_000,
    batch_size: int = 64,
    num_workers: int = 8,
    initial_lr: float = 1e-3,
    lr_decay_rate: float = 4.5e-4,
):
    train_files = get_train_files(dataset_dir)
    val_files = get_val_files(dataset_dir)
    img_categories = get_imagenet_categories(dataset_dir)
    loader = ImageLoader(img_categories=img_categories)

    lr = Tensor(initial_lr)

    @TinyJit
    @MultiModelBase.train()
    def forward_step(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
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
    def get_test_acc(path: Tensor, x: Tensor, y: Tensor) -> Tensor:
        return (
            forward_with_path(marketplace, x, path).argmax(axis=1) == y
        ).mean() * 100

    # with load_with_workers(
    #     loader,
    #     list(map(pathlib.Path, train_files)),
    #     num_worker=num_workers,
    #     shared_memory_enabled=True,
    # ) as generator:
    test_acc = float("nan")
    current_forward_pass = 1

    shuffled_train_files = list(map(pathlib.Path, train_files))
    random.shuffle(shuffled_train_files)

    shuffled_test_files = list(map(pathlib.Path, val_files))
    random.shuffle(shuffled_test_files)

    consumed_count = 0
    generator = load(loader, shuffled_train_files)
    for i in (t := trange(step_count)):
        GlobalCounters.reset()

        start_time = time.perf_counter()

        all_loss = []
        all_paths = []
        for _ in range(current_forward_pass):
            x_batch = []
            y_batch = []
            for _ in range(batch_size):
                x, y = next(generator)
                x_batch.append(x)
                y_batch.append(y)
            x = Tensor.stack(x_batch, dim=0).realize()
            y = Tensor.stack(y_batch, dim=0).realize()

            batch_loss, batch_path = forward_step(x, y)
            all_loss.append(batch_loss)
            all_paths.append(batch_path)
        consumed_count += batch_size * current_forward_pass
        if len(shuffled_train_files) - consumed_count < (
            batch_size * current_forward_pass
        ):
            random.shuffle(shuffled_test_files)
            generator = load(loader, shuffled_train_files)
            consumed_count = 0
            print("Out of training data, reload")

        combined_loss = Tensor.cat(*all_loss).realize()
        combined_paths = Tensor.cat(*all_paths).realize()

        loss, path = mutate_step(
            combined_loss=combined_loss, combined_paths=combined_paths
        )

        end_time = time.perf_counter()
        run_time = end_time - start_time
        lr.replace(lr * (1 - lr_decay_rate))
        gflops = GlobalCounters.global_ops * 1e-9 / run_time

        if i % 10 == (10 - 1):
            # TODO: optimize this
            test_generator = load(loader, shuffled_test_files)

            x_batch = []
            y_batch = []
            # XXX: well, this is not great, but let's some quick hack to make it works
            for _ in range(batch_size * 128):
                x, y = next(test_generator)
                x_batch.append(x)
                y_batch.append(y)

            x = Tensor.stack(x_batch, dim=0).realize()
            y = Tensor.stack(y_batch, dim=0).realize()
            test_acc = get_test_acc(path, x, y).item()

            mlflow.log_metric("training/loss", loss.item(), step=i)
            mlflow.log_metric("training/forward_pass", current_forward_pass, step=i)
            mlflow.log_metric("training/lr", lr.item(), step=i)
            mlflow.log_metric("training/gflops", gflops, step=i)
            mlflow.log_metric("testing/accuracy", test_acc, step=i)

        t.set_description(
            f"loss: {loss.item():6.2f}, fw: {current_forward_pass}, rl: {lr.item():e}, "
            f"acc: {test_acc:5.2f}%, {gflops:9,.2f} GFLOPS"
        )


def main():
    exp_id = ensure_experiment("ResNet18")
    with mlflow.start_run(experiment_id=exp_id, run_name="resnet18"):
        mlflow.log_param("vendor_count", 4)
        mlflow.log_param("num_classes", 10)
        mlflow.log_param("dataset", "mnist")
        train(
            dataset_dir=pathlib.Path("mnist"),
            marketplace=make_marketplace(num_classes=10),
        )


if __name__ == "__main__":
    main()
