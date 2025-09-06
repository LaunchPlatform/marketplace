import logging
import pathlib

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .continual_learning import learn
from .utils import ensure_experiment

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Continual Learning V4")
    checkpoint_file = pathlib.Path(
        "continual-learning-v3-exclude-9-neutral.safetensors"
    )
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
                # exclude 9
                only_classes=tuple(range(9)),
            )
    else:
        logger.info("Checkpoint file %s already exists, skip", checkpoint_file)
    return
    for learn_vendor_count in [4, 8, 16]:
        for fw in [1, 2, 4, 8, 16]:
            for lr in [9e-3, 1e-2, 2e-2, 3e-2, 1e-1, 2e-1]:
                with mlflow.start_run(
                    run_name=f"learn-vendor-{learn_vendor_count}-lr-{lr:.1e}-fw-{fw}",
                    experiment_id=exp_id,
                    log_system_metrics=True,
                ):
                    marketplace = make_marketplace(
                        default_vendor_count=learn_vendor_count
                    )
                    mlflow.log_param("vendor_count", learn_vendor_count)
                    learn(
                        step_count=10_000,
                        batch_size=256,
                        target_new_classes=(3,),
                        new_train_size=32,
                        initial_lr=lr,
                        lr_decay_rate=0,
                        probe_scale=1.0,
                        forward_pass=fw,
                        marketplace=marketplace,
                        manual_seed=42,
                        input_checkpoint_filepath=checkpoint_file,
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
