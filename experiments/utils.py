import logging
import typing

import mlflow
import numpy as np
from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.tensor import ReductionStr

logger = logging.getLogger(__name__)


def ensure_experiment(name: str) -> str:
    try:
        experiment_id = mlflow.create_experiment(
            name=name,
        )
        logger.info("Created experiment with name %s and id %s", name, experiment_id)
        return experiment_id
    except mlflow.exceptions.MlflowException as e:
        logger.info("Failed to create experiment with error: %s", e)
        # If experiment already exists, get its ID
        experiment = mlflow.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id
        logger.info("Return existing experiment id %s for %s", experiment_id, name)
        return experiment_id


def filter_classes(
    x: Tensor, y: Tensor, only: typing.Container
) -> tuple[Tensor, Tensor]:
    class_mask = np.isin(y.numpy(), only)
    return Tensor(x.numpy()[class_mask]), Tensor(y.numpy()[class_mask])


# We copy sparse_categorical_crossentropy from tinygrad and add our own neutral mask to make the model output
# neutral for reserved labels
# ref: https://github.com/tinygrad/tinygrad/blob/35ddfc3d39cbf7bc0ee3d17331788c02e031508e/tinygrad/tensor.py#L3987-L4010
def sparse_categorical_crossentropy_with_neutral_mask(
    self,
    Y: Tensor,
    ignore_index: int = -1,
    neutral_mask: Tensor | None = None,
    label_smoothing=0.0,
    reduction: ReductionStr = "mean",
) -> Tensor:
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    assert reduction in typing.get_args(ReductionStr), (
        f"reduction must be one of {typing.get_args(ReductionStr)}"
    )
    log_probs = self.log_softmax()
    loss_mask = (
        (Y != ignore_index) if ignore_index != -1 else Y.ones_like(dtype=dtypes.bool)
    )
    y = Y.to(self.device).unsqueeze(-1)._one_hot_along_dim(
        self.shape[-1], dim=-1
    ) * loss_mask.unsqueeze(-1)
    if neutral_mask is not None:
        y += neutral_mask
    smoothing = label_smoothing * (log_probs.mean(-1) * loss_mask)
    unreduced = (1 - label_smoothing) * (log_probs * y).sum(-1) + smoothing
    # NOTE: because of ignore_index, we can't use Tensor.mean (so can't use `_do_reduction` here)
    return -(
        unreduced.sum() / loss_mask.sum()
        if reduction == "mean"
        else (unreduced.sum() if reduction == "sum" else unreduced)
    )
