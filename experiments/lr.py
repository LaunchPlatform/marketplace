import logging

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Learning Rate V5")
    for lr in [1e-2, 1e-3, 1e-4]:
        for decay in [1e-3, 1e-4, 1e-5]:
            with mlflow.start_run(
                run_name=f"bs-512-lr-{lr:.1e}-decay-{decay:.1e}",
                experiment_id=exp_id,
                log_system_metrics=True,
            ):
                marketplace = make_marketplace(default_vendor_count=8)
                train(
                    step_count=3_000,
                    batch_size=512,
                    initial_lr=lr,
                    lr_decay_rate=decay,
                    marketplace=marketplace,
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
