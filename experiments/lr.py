import logging

import mlflow

from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Learning Rate")
    for lr, decay in [
        (1e-3, 3.0e-4),
        (1e-3, 3.5e-4),
        (1e-3, 4.0e-4),
        (1e-3, 4.5e-4),
        (1e-3, 5.0e-4),
    ]:
        with mlflow.start_run(
            run_name=f"lr-{lr:.1e}-decay-{decay:.1e}-round-2",
            experiment_id=exp_id,
            description="Find out how learning rate and decay rate affects the training process",
            log_system_metrics=True,
        ):
            train(
                step_count=10_000,
                batch_size=32,
                initial_lr=lr,
                lr_decay_rate=decay,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
