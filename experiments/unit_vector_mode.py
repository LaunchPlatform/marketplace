import logging

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment
from marketplace.optimizers import UnitVectorMode

logger = logging.getLogger(__name__)

VENDOR_COUNT = 8


def main():
    exp_id = ensure_experiment("Unit Vector Mode")
    for lr in [0.09, 0.1, 0.2, 0.3]:
        for mode in [UnitVectorMode.per_spec, UnitVectorMode.whole]:
            with mlflow.start_run(
                run_name=f"lr-{lr}-{mode.value}-scale-1e-3",
                experiment_id=exp_id,
                log_system_metrics=True,
            ):
                marketplace = make_marketplace(default_vendor_count=VENDOR_COUNT)
                mlflow.log_param("vendor_count", VENDOR_COUNT)
                train(
                    step_count=1_000,
                    batch_size=512,
                    initial_lr=lr,
                    lr_decay_rate=1e-5,
                    probe_scale=0.001,
                    unit_vector_mode=mode,
                    marketplace=marketplace,
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
