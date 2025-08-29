import logging

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("LR Scale")
    for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
        for lr_scale_start in [1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1]:
            for lr_scale_end in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for decay in [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                    with mlflow.start_run(
                        run_name=f"lr-scale-lr-{lr:.1e}-scale-{lr_scale_start:.1e}-{lr_scale_end:.1e}-decay-{decay:.1e}",
                        experiment_id=exp_id,
                        log_system_metrics=True,
                    ):
                        marketplace = make_marketplace(default_vendor_count=8)
                        mlflow.log_param("vendor_count", 8)
                        train(
                            step_count=1_000,
                            batch_size=512,
                            initial_lr=lr,
                            lr_decay_rate=decay,
                            lr_scaling_range=(lr_scale_end, lr_scale_start),
                            marketplace=marketplace,
                        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
