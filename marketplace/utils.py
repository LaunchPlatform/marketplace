import itertools
import logging
import pathlib

from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import safe_save

from .training import Spec

logger = logging.getLogger(__name__)


def write_checkpoint(
    marketplace: list[Spec],
    path: Tensor,
    global_step: int,
    output_filepath: pathlib.Path,
):
    logger.info(
        "Writing checkpoint with global_step %s to %s", global_step, output_filepath
    )
    parameters = dict(
        itertools.chain.from_iterable(
            [
                (f"layer.{i}.{key}", weights[index])
                for key, weights in get_state_dict(spec.model).items()
            ]
            for i, (index, spec) in enumerate(zip(path, marketplace))
        )
    )
    checkpoint_tmp_filepath = output_filepath.with_suffix(".tmp")
    safe_save(
        parameters | dict(global_step=Tensor(global_step)), str(checkpoint_tmp_filepath)
    )
    checkpoint_tmp_filepath.rename(output_filepath)
    logger.info(
        "Wrote checkpoint with global_step %s to %s", global_step, output_filepath
    )
