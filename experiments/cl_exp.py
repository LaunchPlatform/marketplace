import logging
import pathlib

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Continual Learning")
    checkpoint_file = pathlib.Path("continual-learning.safetensors")
    if not checkpoint_file.exists():
        train_vendor_count = 16
        with mlflow.start_run(
            run_name="train-base-model",
            experiment_id=exp_id,
            log_system_metrics=True,
        ):
            marketplace = make_marketplace(default_vendor_count=train_vendor_count)
            mlflow.log_param("vendor_count", train_vendor_count)
            train(
                step_count=2_000,
                batch_size=512,
                initial_lr=1e-1,
                lr_decay_rate=1e-5,
                probe_scale=1e-1,
                marketplace=marketplace,
                manual_seed=42,
                checkpoint_filepath=checkpoint_file,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
