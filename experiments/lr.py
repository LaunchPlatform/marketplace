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
    # layer 2
    (4, 4),
    # layer 3
    (8, 4),
    # layer 4
    (16, 8),
    # layer 5
    (16, 16),
    # layer 6
    (32, 16),
]


def main():
    exp_id = ensure_experiment("Learning Rate V2")
    for batch_size in [32, 64, 128, 256, 512]:
        for lr in [1e-2, 1e-3, 1e-4]:
            for decay in [1e-3, 1e-4, 1e-5]:
                with mlflow.start_run(
                    run_name=f"bs-{batch_size}-lr-{lr:.1e}-decay-{decay:.1e}",
                    experiment_id=exp_id,
                    log_system_metrics=True,
                ):
                    marketplace = make_marketplace(PYRAMID32_HALF_UPSTREAM_STRUCTURE)
                    train(
                        step_count=3_000,
                        batch_size=batch_size,
                        initial_lr=lr,
                        lr_decay_rate=decay,
                        marketplace=marketplace,
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
