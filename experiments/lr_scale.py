import logging

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("LR Scale")
    for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
        for lr_scale_start in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for lr_scale_end in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for decay in [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                    with mlflow.start_run(
                        run_name=f"lr-scale-lr-{lr:.1e}-scale-{lr_scale_start}-{lr_scale_end}-decay-{decay:.1e}",
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
                            lr_scaling_range=(lr_scale_start, lr_scale_end),
                            marketplace=marketplace,
                        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
