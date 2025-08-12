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
    exp_id = ensure_experiment("Long Run")
    with mlflow.start_run(
        run_name=f"long-run-v2",
        experiment_id=exp_id,
        description="Find out how learning rate and decay rate affects the training process",
        log_system_metrics=True,
        tags=dict(round="5"),
    ):
        marketplace = make_marketplace(PYRAMID32_HALF_UPSTREAM_STRUCTURE)
        train(
            step_count=100_000,
            batch_size=32,
            initial_lr=1e-3,
            lr_decay_rate=1e-3,
            marketplace=marketplace,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
