import logging

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .utils import ensure_experiment
from marketplace.multi_nn import MultiBatchNorm
from marketplace.multi_nn import MultiInstanceNorm

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Batch Normal vs Instance Normal")
    for batch_normal in [True, False]:
        with mlflow.start_run(
            run_name="batch-normal" if batch_normal else "instance-normal",
            experiment_id=exp_id,
            log_system_metrics=True,
        ):
            marketplace = make_marketplace(
                norm_cls=MultiBatchNorm if batch_normal else MultiInstanceNorm
            )
            train(
                step_count=10_000,
                batch_size=512,
                initial_forward_pass=1,
                initial_lr=1e-3,
                lr_decay_rate=1e-4,
                marketplace=marketplace,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
