import logging
import sys

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Scaling V2")
    for marketplace_replica in [  # 1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]:
        for forward_pass in [1, 2, 4, 8, 16, 32, 64]:
            with mlflow.start_run(
                run_name=f"scaling-mr-{marketplace_replica}-fw-{forward_pass}",
                experiment_id=exp_id,
                log_system_metrics=True,
            ):
                marketplace = make_marketplace(default_vendor_count=8)
                train(
                    step_count=3_000,
                    batch_size=512,
                    initial_lr=1e-3,
                    lr_decay_rate=1e-4,
                    initial_forward_pass=forward_pass,
                    marketplace=marketplace,
                    marketplace_replica=marketplace_replica,
                    # Make initial weights the same so that the exp is less noisy
                    manual_seed=42,
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ref: https://github.com/tinygrad/tinygrad/issues/8617
    # With complex huge compute graph, tinygrad runs into recursion too deep issue, let's bump it up
    NEW_RECURSION_LIMIT = 100_000
    logger.info("Current recursion limit is %s", sys.getrecursionlimit())
    sys.setrecursionlimit(NEW_RECURSION_LIMIT)
    logger.info("Set recursion limit to %s", NEW_RECURSION_LIMIT)

    main()
