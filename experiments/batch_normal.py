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
    exp_id = ensure_experiment("Batch Normal")
    for batch_size, batch_normal, evolve in [
        # (32, False, False),
        # (32, True, False),
        (32, True, True),
        # (64, False, False),
        # (64, True, False),
        (64, True, True),
        # (128, False, False),
        # (128, True, False),
        (128, True, True),
        # (256, False, False),
        # (256, True, False),
        (256, True, True),
        # (512, False, False),
        # (512, True, False),
        (512, True, True),
    ]:
        with mlflow.start_run(
            run_name=f"normal-batch-{batch_size}-nb-{batch_normal}-evolve-{evolve}",
            experiment_id=exp_id,
            log_system_metrics=True,
        ):
            marketplace = make_marketplace(
                PYRAMID32_HALF_UPSTREAM_STRUCTURE,
                batch_normal=batch_normal,
                evolve_batch_normal=evolve,
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
