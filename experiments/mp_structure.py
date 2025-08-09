import logging

import mlflow

from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)

HOMOGENEOUS_HALF_UPSTREAM_STRUCTURE = [
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
HOMOGENEOUS_FULL_UPSTREAM_STRUCTURE = [
    # layer 0
    (32, 0),
    # layer 1
    (32, 32),
    # layer 2 (N/A)
    (0, 0),
    # layer 3
    (32, 32),
    # layer 4
    (32, 32),
    # layer 5 (N/A)
    (0, 0),
    # layer 6
    (32, 32),
]
PYRAMID64_HALF_UPSTREAM_STRUCTURE = [
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
PYRAMID64_FULL_UPSTREAM_STRUCTURE = [
    # layer 0
    (4, 0),
    # layer 1
    (8, 4),
    # layer 2 (N/A)
    (0, 0),
    # layer 3
    (16, 16),
    # layer 4
    (32, 32),
    # layer 5 (N/A)
    (0, 0),
    # layer 6
    (64, 64),
]
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
PYRAMID32_FULL_UPSTREAM_STRUCTURE = [
    # layer 0
    (2, 0),
    # layer 1
    (4, 2),
    # layer 2 (N/A)
    (0, 0),
    # layer 3
    (8, 8),
    # layer 4
    (16, 16),
    # layer 5 (N/A)
    (0, 0),
    # layer 6
    (32, 32),
]


def main():
    exp_id = ensure_experiment("Marketplace Structure")
    for name, forward_pass, structure in [
        # ("homogeneous-half", 1, HOMOGENEOUS_HALF_UPSTREAM_STRUCTURE),
        # ("homogeneous-half", 2, HOMOGENEOUS_HALF_UPSTREAM_STRUCTURE),
        # ("homogeneous-half", 4, HOMOGENEOUS_HALF_UPSTREAM_STRUCTURE),
        # ("homogeneous-half", 8, HOMOGENEOUS_HALF_UPSTREAM_STRUCTURE),
        # ("homogeneous-full", 1, HOMOGENEOUS_FULL_UPSTREAM_STRUCTURE),
        # ("homogeneous-full", 2, HOMOGENEOUS_FULL_UPSTREAM_STRUCTURE),
        # ("homogeneous-full", 4, HOMOGENEOUS_FULL_UPSTREAM_STRUCTURE),
        # ("homogeneous-full", 8, HOMOGENEOUS_FULL_UPSTREAM_STRUCTURE),
        ("pyramid64-half", 1, PYRAMID64_HALF_UPSTREAM_STRUCTURE),
        ("pyramid64-half", 2, PYRAMID64_HALF_UPSTREAM_STRUCTURE),
        ("pyramid64-half", 4, PYRAMID64_HALF_UPSTREAM_STRUCTURE),
        ("pyramid64-half", 8, PYRAMID64_HALF_UPSTREAM_STRUCTURE),
        ("pyramid64-full", 1, PYRAMID64_FULL_UPSTREAM_STRUCTURE),
        ("pyramid64-full", 2, PYRAMID64_FULL_UPSTREAM_STRUCTURE),
        ("pyramid64-full", 4, PYRAMID64_FULL_UPSTREAM_STRUCTURE),
        ("pyramid64-full", 8, PYRAMID64_FULL_UPSTREAM_STRUCTURE),
        ("pyramid32-half", 1, PYRAMID32_HALF_UPSTREAM_STRUCTURE),
        ("pyramid32-half", 2, PYRAMID32_HALF_UPSTREAM_STRUCTURE),
        ("pyramid32-half", 4, PYRAMID32_HALF_UPSTREAM_STRUCTURE),
        ("pyramid32-half", 8, PYRAMID32_HALF_UPSTREAM_STRUCTURE),
        ("pyramid32-full", 1, PYRAMID32_FULL_UPSTREAM_STRUCTURE),
        ("pyramid32-full", 2, PYRAMID32_FULL_UPSTREAM_STRUCTURE),
        ("pyramid32-full", 4, PYRAMID32_FULL_UPSTREAM_STRUCTURE),
        ("pyramid32-full", 8, PYRAMID32_FULL_UPSTREAM_STRUCTURE),
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
                initial_forward_pass=forward_pass,
                mp_structure=structure,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
