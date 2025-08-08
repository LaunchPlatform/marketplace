import logging

import mlflow

from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)

HOMOGENEOUS_STRUCTURE = [
    # layer 0
    (32, 0),
    # layer 1
    (32, 16),
    # layer 2 (N/A)
    (0, 0),
    # layer 3
    (32, 16),
    # layer 4
    (32, 16),
    # layer 5 (N/A)
    (0, 0),
    # layer 6
    (32, 16),
]
PYRAMID_STRUCTURE = [
    # layer 0
    (4, 0),
    # layer 1
    (8, 4),
    # layer 2 (N/A)
    (0, 0),
    # layer 3
    (16, 8),
    # layer 4
    (32, 16),
    # layer 5 (N/A)
    (0, 0),
    # layer 6
    (64, 32),
]


def main():
    exp_id = ensure_experiment("Marketplace Structure")
    for name, forward_pass, structure in [
        ("homogeneous", 1, HOMOGENEOUS_STRUCTURE),
        ("homogeneous", 2, HOMOGENEOUS_STRUCTURE),
        ("homogeneous", 4, HOMOGENEOUS_STRUCTURE),
        ("homogeneous", 8, HOMOGENEOUS_STRUCTURE),
        ("pyramid", 1, PYRAMID_STRUCTURE),
        ("pyramid", 2, PYRAMID_STRUCTURE),
        ("pyramid", 4, PYRAMID_STRUCTURE),
        ("pyramid", 8, PYRAMID_STRUCTURE),
    ]:
        with mlflow.start_run(
            run_name=f"{name}-fw{forward_pass}",
            experiment_id=exp_id,
            description="Find out how marketplace structure impacts the performance of training",
            log_system_metrics=True,
            tags=dict(round="0"),
        ):
            train(
                step_count=3_000,
                batch_size=32,
                initial_lr=1e-3,
                lr_decay_rate=4.5e-4,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
