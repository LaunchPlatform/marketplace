import logging

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Forward Pass")
    for forward_pass in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        with mlflow.start_run(
            run_name=f"forward-pass-{forward_pass}",
            experiment_id=exp_id,
            log_system_metrics=True,
        ):
            marketplace = make_marketplace(default_vendor_count=12)
            train(
                step_count=10_000,
                batch_size=512,
                initial_lr=1e-3,
                lr_decay_rate=1e-4,
                initial_forward_pass=forward_pass,
                marketplace=marketplace,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
