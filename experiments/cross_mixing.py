import logging
import sys

import mlflow

from .beautiful_mnist import make_deep_marketplace
from .beautiful_mnist import make_marketplace
from .beautiful_mnist import make_marketplace_without_cross_mixing
from .beautiful_mnist import train
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Cross Mixing V3")
    for cross_mixing, vendor_count in [
        # (True, 8),
        # (True, 16),
        # (False, 32),
        # (False, 64),
        (True, 8)
    ]:
        with mlflow.start_run(
            run_name=f"cross-mixing-deep-market-{vendor_count}",
            experiment_id=exp_id,
            description="Find out if cross mixing indeed helpful or not",
            log_system_metrics=True,
        ):
            if True:
                marketplace = make_deep_marketplace(default_vendor_count=vendor_count)
            elif cross_mixing:
                marketplace = make_marketplace(default_vendor_count=vendor_count)
            else:
                marketplace = make_marketplace_without_cross_mixing(vendor_count)
            train(
                step_count=10_000,
                batch_size=512,
                initial_lr=1e-3,
                lr_decay_rate=4.5e-4,
                marketplace=marketplace,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ref: https://github.com/tinygrad/tinygrad/issues/8617
    # With complex huge compute graph, tinygrad runs into recursion too deep issue, let's bump it up
    NEW_RECURSION_LIMIT = 100_000
    logger.info("Current recursion limit is %s", sys.getrecursionlimit())
    logger.info("Set recursion limit to %s", NEW_RECURSION_LIMIT)
    sys.setrecursionlimit(NEW_RECURSION_LIMIT)
    main()
