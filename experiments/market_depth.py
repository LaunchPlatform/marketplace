import logging
import sys
import typing

import mlflow
from tinygrad import Tensor

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment
from marketplace.multi_nn import MultiConv2d
from marketplace.multi_nn import MultiInstanceNorm
from marketplace.multi_nn import MultiLinear
from marketplace.multi_nn import MultiModel
from marketplace.multi_nn import MultiModelBase
from marketplace.training import Spec

logger = logging.getLogger(__name__)


def make_marketplace_depth_1(
    vendor_count: int,
    norm_cls: typing.Type[MultiModelBase] = MultiInstanceNorm,
):
    return [
        Spec(
            model=MultiModel(
                [
                    MultiConv2d(vendor_count, 1, 32, 5),
                    Tensor.relu,
                    MultiConv2d(vendor_count, 32, 32, 5),
                    Tensor.relu,
                    norm_cls(vendor_count, 32),
                    Tensor.max_pool2d,
                    MultiConv2d(vendor_count, 32, 64, 3),
                    Tensor.relu,
                    MultiConv2d(vendor_count, 64, 64, 3),
                    Tensor.relu,
                    norm_cls(vendor_count, 64),
                    Tensor.max_pool2d,
                    lambda x: x.flatten(1),
                    MultiLinear(vendor_count, 576, 10),
                ]
            ),
        ),
    ]


def main():
    exp_id = ensure_experiment("Market Depth V2")
    for market_depth, vendor_count in [
        (1, 8),
        (1, 16),
        (1, 32),
        (1, 64),
        (3, 8),
        (3, 16),
    ]:
        with mlflow.start_run(
            run_name=f"market-depth-{market_depth}-vendor-{vendor_count}",
            experiment_id=exp_id,
            description="Find out how market depth affects performance",
            log_system_metrics=True,
        ):
            if market_depth == 3:
                marketplace = make_marketplace(default_vendor_count=vendor_count)
            elif market_depth == 1:
                marketplace = make_marketplace_depth_1(vendor_count)
            else:
                raise ValueError(f"Unexpected depth {market_depth}")
            train(
                step_count=10_000,
                batch_size=512,
                initial_lr=1e-3,
                lr_decay_rate=1e-4,
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
