# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import logging
import os
import time
from typing import Callable

import mlflow
from tinygrad import dtypes
from tinygrad import GlobalCounters
from tinygrad import nn
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import colored
from tinygrad.helpers import getenv
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import load_state_dict
from tinygrad.nn.state import safe_load

from experiments.beautiful_mnist import make_marketplace
from experiments.beautiful_mnist import train
from experiments.utils import ensure_experiment

DEPTH_3_MODEL_STATE_KEY_MAP = {
    "spec.0.layers.0.bias": "layers.0.bias",
    "spec.0.layers.0.weight": "layers.0.weight",
    "spec.0.layers.2.bias": "layers.2.bias",
    "spec.0.layers.2.weight": "layers.2.weight",
    "spec.0.layers.4.bias": "layers.4.bias",
    "spec.0.layers.4.weight": "layers.4.weight",
    "spec.0.layers.4.running_mean": "layers.4.running_mean",
    "spec.0.layers.4.running_var": "layers.4.running_var",
    "spec.1.layers.0.bias": "layers.6.bias",
    "spec.1.layers.0.weight": "layers.6.weight",
    "spec.1.layers.2.bias": "layers.8.bias",
    "spec.1.layers.2.weight": "layers.8.weight",
    "spec.1.layers.4.bias": "layers.10.bias",
    "spec.1.layers.4.weight": "layers.10.weight",
    "spec.1.layers.4.running_mean": "layers.10.running_mean",
    "spec.1.layers.4.running_var": "layers.10.running_var",
    "spec.2.layers.0.bias": "layers.13.bias",
    "spec.2.layers.0.weight": "layers.13.weight",
}
BATCH_NORM_KEYS = ["layers.4", "layers.10"]
logger = logging.getLogger(__name__)


class Model:
    def __init__(self):
        self.layers: list[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5),
            Tensor.relu,
            nn.Conv2d(32, 32, 5),
            Tensor.relu,
            nn.InstanceNorm(32),
            Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3),
            Tensor.relu,
            nn.Conv2d(64, 64, 3),
            Tensor.relu,
            nn.InstanceNorm(64),
            Tensor.max_pool2d,
            lambda x: x.flatten(1),
            nn.Linear(576, 10),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


def train_mnist(optimizer_type: str = "adam", lr: float = 1e-3):
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

    model = Model()

    model_weights_filepath = os.environ.get("MODEL_WEIGHTS")
    if model_weights_filepath is not None:
        logger.info("Loading model weights from %s", model_weights_filepath)
        model_state = safe_load(model_weights_filepath)
        converted_state = {
            DEPTH_3_MODEL_STATE_KEY_MAP[key]: value
            for key, value in model_state.items()
            if key in DEPTH_3_MODEL_STATE_KEY_MAP
        }
        for key in BATCH_NORM_KEYS:
            converted_state[f"{key}.num_batches_tracked"] = Tensor.zeros(
                1,
                dtype="long" if is_dtype_supported(dtypes.long) else "int",
                requires_grad=False,
            )
        load_state_dict(model, converted_state)
        logger.info("Model weight loaded")

    if optimizer_type == "adam":
        opt = nn.optim.Adam(nn.state.get_parameters(model))
        mlflow.log_param("optimizer", "adam")
    elif optimizer_type == "muon":
        opt = nn.optim.Muon(nn.state.get_parameters(model), lr=lr)
        mlflow.log_param("lr", lr)
    elif optimizer_type == "sgd":
        opt = nn.optim.SGD(
            nn.state.get_parameters(model),
            lr=lr,
        )
        mlflow.log_param("lr", lr)
    else:
        raise ValueError("Unexpected type")
    mlflow.log_param("optimizer", optimizer_type)

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
        loss = (
            model(X_train[samples])
            .sparse_categorical_crossentropy(Y_train[samples])
            .backward()
        )
        opt.step()
        return loss

    @TinyJit
    def get_test_acc() -> Tensor:
        return (model(X_test).argmax(axis=1) == Y_test).mean() * 100

    test_acc = float("nan")
    for i in (t := trange(getenv("STEPS", 3_000))):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        loss = train_step()
        start_time = time.perf_counter()
        run_time = time.perf_counter() - start_time
        gflops = GlobalCounters.global_ops * 1e-9 / run_time
        if i % 10 == 9:
            test_acc = get_test_acc().item()
            mlflow.log_metric("training/loss", loss.item(), step=i)
            mlflow.log_metric("training/gflops", gflops, step=i)
            mlflow.log_metric("testing/accuracy", test_acc, step=i)
        t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

    # verify eval acc
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))


if __name__ == "__main__":
    exp_id = ensure_experiment("Backprop Comparison V3")
    for optimizer_type in ["adam", "muon", "sgd"]:
        if optimizer_type == "adam":
            with mlflow.start_run(
                run_name=f"backprop-adam",
                experiment_id=exp_id,
                log_system_metrics=True,
            ):
                train_mnist(optimizer_type=optimizer_type)
        else:
            for lr_base in [1e-2, 1e-3, 1e-4]:
                for lr in list(map(lambda x: x * 1e-3, range(1, 10))):
                    with mlflow.start_run(
                        run_name=f"backprop-{optimizer_type}-lr-{lr}",
                        experiment_id=exp_id,
                        log_system_metrics=True,
                    ):
                        train_mnist(optimizer_type=optimizer_type, lr=lr)
    with mlflow.start_run(
        run_name="marketplace",
        experiment_id=exp_id,
        log_system_metrics=True,
    ):
        marketplace = make_marketplace(default_vendor_count=16)
        train(
            step_count=3_000,
            batch_size=512,
            initial_lr=1e-3,
            lr_decay_rate=1e-4,
            marketplace=marketplace,
        )
