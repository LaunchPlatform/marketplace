import logging
import pathlib

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .continual_learning_fashion import learn
from .utils import ensure_experiment
from marketplace.optimizers import UnitVectorMode

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Continual Learning Fashion - Article")
    checkpoint_file = pathlib.Path("continual-learning-fashion.safetensors")
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
                initial_lr=3e-1,
                lr_decay_rate=1e-5,
                probe_scale=1e-2,
                marketplace=marketplace,
                manual_seed=42,
                unit_vector_mode=UnitVectorMode.whole,
                checkpoint_filepath=checkpoint_file,
            )
    else:
        logger.info("Checkpoint file %s already exists, skip", checkpoint_file)

    learn_vendor_count = 4
    for fw in [4, 8, 16]:
        for lr in [1e-3, 5e-3, 1e-2, 2e-2, 3e-2]:
            for probe_scale in [1, 0.5, 0.1]:
                with mlflow.start_run(
                    run_name=f"learn--lr-{lr:.1e}-fw-{fw}-probe-scale-{probe_scale}",
                    experiment_id=exp_id,
                    log_system_metrics=True,
                ):
                    marketplace = make_marketplace(
                        default_vendor_count=learn_vendor_count,
                    )
                    mlflow.log_param("vendor_count", learn_vendor_count)
                    learn(
                        step_count=10_000,
                        batch_size=256,
                        new_train_size=16,
                        initial_lr=lr,
                        lr_decay_rate=0,
                        probe_scale=probe_scale,
                        forward_pass=fw,
                        marketplace=marketplace,
                        manual_seed=42,
                        input_checkpoint_filepath=checkpoint_file,
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
