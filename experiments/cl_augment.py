import logging
import pathlib

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .continual_learning import learn
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Continual Learning Augment")
    checkpoint_file = pathlib.Path("continual-learning.safetensors")
    if not checkpoint_file.exists():
        train_vendor_count = 16
        with mlflow.start_run(
            run_name="base-model",
            experiment_id=exp_id,
            log_system_metrics=True,
        ):
            marketplace = make_marketplace(default_vendor_count=train_vendor_count)
            mlflow.log_param("vendor_count", train_vendor_count)
            train(
                step_count=2_000,
                batch_size=512,
                initial_lr=1e-1,
                lr_decay_rate=1e-5,
                probe_scale=1e-1,
                marketplace=marketplace,
                manual_seed=42,
                checkpoint_filepath=checkpoint_file,
            )
    else:
        logger.info("Checkpoint file %s already exists, skip", checkpoint_file)
    learn_vendor_count = 4
    for balance_labels in [True, False]:
        with mlflow.start_run(
            run_name=f"balance-labels-{balance_labels}",
            experiment_id=exp_id,
            log_system_metrics=True,
        ):
            marketplace = make_marketplace(default_vendor_count=learn_vendor_count)
            mlflow.log_param("vendor_count", learn_vendor_count)
            learn(
                step_count=10_000,
                batch_size=256,
                target_new_classes=(3,),
                balance_labels=balance_labels,
                new_train_size=32,
                initial_lr=1e-2,
                lr_decay_rate=0,
                probe_scale=1.0,
                forward_pass=1,
                marketplace=marketplace,
                manual_seed=42,
                input_checkpoint_filepath=checkpoint_file,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
