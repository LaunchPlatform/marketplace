import logging

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)

PYRAMID32_HALF_UPSTREAM_STRUCTURE = [
    # layer 0
    (2, 0),
    # layer 1
    (4, 2),
    # layer 2 (N/A)
    (0, 0),
    # layer 3
    (8, 4),
    # layer 4
    (16, 8),
    # layer 5 (N/A)
    (0, 0),
    # layer 6
    (32, 16),
]


def main():
    exp_id = ensure_experiment("Batch Size")
    for batch_size, forward_pass in [
        # (32, 1),
        (32, 2),
        (32, 4),
        (32, 8),
        (32, 16),
        # (64, 1),
        (64, 2),
        (64, 4),
        (64, 8),
        # (128, 1),
        (128, 2),
        (128, 4),
        # (256, 1),
        (256, 2),
        # (512, 1),
    ]:
        with mlflow.start_run(
            run_name=f"batch-size-{batch_size}-fw-{forward_pass}",
            experiment_id=exp_id,
            log_system_metrics=True,
        ):
            marketplace = make_marketplace(PYRAMID32_HALF_UPSTREAM_STRUCTURE)
            train(
                step_count=10_000,
                batch_size=batch_size,
                initial_forward_pass=forward_pass,
                initial_lr=1e-3,
                lr_decay_rate=1e-4,
                marketplace=marketplace,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
