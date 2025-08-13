import logging

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import make_marketplace_without_cross_mixing
from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)

PYRAMID32_HALF_UPSTREAM_STRUCTURE = [
    # layer 0
    (2, 0),
    # layer 1
    (4, 2),
    # layer 2 (N/A)
    (4, 4),
    # layer 3
    (8, 4),
    # layer 4
    (16, 8),
    # layer 5 (N/A)
    (16, 16),
    # layer 6
    (32, 16),
]


def main():
    exp_id = ensure_experiment("Cross Mixing V2")
    for cross_mixing, vendor_count, sticky_leaders in [
        # (False, 32, False),
        # (False, 64, False),
        # (False, 128, False),
        # (True, 0, False),
        (True, 0, True),
        # (True, 0, False),
    ]:
        with mlflow.start_run(
            run_name=f"cross-mixing-{cross_mixing}-{vendor_count}-sticky-leader-{sticky_leaders}-round2",
            experiment_id=exp_id,
            description="Find out if cross mixing indeed helpful or not",
            log_system_metrics=True,
        ):
            if cross_mixing:
                marketplace = make_marketplace(PYRAMID32_HALF_UPSTREAM_STRUCTURE)
            else:
                marketplace = make_marketplace_without_cross_mixing(vendor_count)
            train(
                step_count=3_000,
                batch_size=512,
                initial_lr=1e-3,
                lr_decay_rate=4.5e-4,
                sticky_leaders=sticky_leaders,
                marketplace=marketplace,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
