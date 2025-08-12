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
    exp_id = ensure_experiment("Batch Normal")
    for batch_size, batch_normal in [
        (32, False),
        (32, True),
        (64, False),
        (64, True),
        (128, False),
        (128, True),
        (256, False),
        (256, True),
        (512, False),
        (512, True),
    ]:
        with mlflow.start_run(
            run_name=f"normal-batch-{batch_size}-nb-{batch_normal}",
            experiment_id=exp_id,
            log_system_metrics=True,
        ):
            marketplace = make_marketplace(
                PYRAMID32_HALF_UPSTREAM_STRUCTURE, batch_normal=batch_normal
            )
            train(
                step_count=10_000,
                batch_size=batch_size,
                initial_forward_pass=1,
                initial_lr=1e-3,
                lr_decay_rate=1e-4,
                marketplace=marketplace,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
