import logging

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)

VENDOR_COUNT = 8


def main():
    exp_id = ensure_experiment("Param Attribution LR V2")
    for probe in [None, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        for lr in [0.1, 0.5, 1.0]:
            probe_str = f"{probe:.1e}" if probe is not None else "none"
            with mlflow.start_run(
                run_name=f"probe-{probe_str}-lr-{lr:.1e}",
                experiment_id=exp_id,
                log_system_metrics=True,
            ):
                marketplace = make_marketplace(default_vendor_count=VENDOR_COUNT)
                mlflow.log_param("vendor_count", VENDOR_COUNT)
                train(
                    step_count=1_000,
                    batch_size=512,
                    initial_lr=lr,
                    lr_decay_rate=1e-4,
                    probe=probe,
                    marketplace=marketplace,
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
