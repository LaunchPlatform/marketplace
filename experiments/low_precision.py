import logging

import mlflow
from tinygrad import Context

from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


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


def main():
    exp_id = ensure_experiment("Low Precision")
    for fp16 in [
        False,
        True,
    ]:
        with mlflow.start_run(
            run_name=f"fp-16-{fp16}",
            experiment_id=exp_id,
            description="Find out if low precision training make any difference",
            log_system_metrics=True,
            tags=dict(round="5"),
        ):
            ctx_values = {}
            if fp16:
                ctx_values["FLOAT16"] = 1
            with Context(**ctx_values):
                mlflow.log_param("fp16", fp16)
                train(
                    step_count=10_000,
                    batch_size=32,
                    initial_lr=1e-3,
                    lr_decay_rate=4.5e-4,
                    mp_structure=PYRAMID64_HALF_UPSTREAM_STRUCTURE,
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
