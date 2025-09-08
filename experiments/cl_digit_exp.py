import logging
import pathlib

import mlflow

from .beautiful_mnist import make_marketplace
from .beautiful_mnist import train
from .continual_learning import learn
from .utils import ensure_experiment
from marketplace.optimizers import UnitVectorMode

logger = logging.getLogger(__name__)


def main():
    exp_id = ensure_experiment("Continual Learning Digit - Article")
    for exclude_missing_from_loss_func in [False, True]:
        checkpoint_file = pathlib.Path(
            f"continual-learning-v3-exclude-9-exclude-loss-{exclude_missing_from_loss_func}.safetensors"
        )
        if not checkpoint_file.exists():
            train_vendor_count = 16
            with mlflow.start_run(
                run_name=f"base-model-exclude-loss-{exclude_missing_from_loss_func}",
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
                    # exclude 9
                    only_classes=tuple(range(9)),
                    exclude_missing_from_loss_func=exclude_missing_from_loss_func,
                )
        else:
            logger.info("Checkpoint file %s already exists, skip", checkpoint_file)

        learn_vendor_count = 4
        replay_file = pathlib.Path(
            f"digit-exclude-loss-{exclude_missing_from_loss_func}.jsonl"
        )
        with (
            mlflow.start_run(
                run_name=f"exclude-loss-{exclude_missing_from_loss_func}",
                experiment_id=exp_id,
                log_system_metrics=True,
            ),
            replay_file.open("wt") as fo,
        ):
            marketplace = make_marketplace(
                default_vendor_count=learn_vendor_count,
            )
            mlflow.log_param("vendor_count", learn_vendor_count)
            learn(
                step_count=100_000,
                batch_size=256,
                initial_lr=1e-2,
                lr_decay_rate=0,
                probe_scale=1.0,
                forward_pass=1,
                marketplace=marketplace,
                manual_seed=42,
                replay_file=fo,
                input_checkpoint_filepath=checkpoint_file,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
