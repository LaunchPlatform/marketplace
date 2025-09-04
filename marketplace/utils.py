import itertools
import logging
import pathlib

from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import load_state_dict
from tinygrad.nn.state import safe_load
from tinygrad.nn.state import safe_save

from .training import Spec

logger = logging.getLogger(__name__)


def write_checkpoint(
    marketplace: list[Spec],
    global_step: int,
    output_filepath: pathlib.Path,
):
    logger.info(
        "Writing checkpoint with global_step %s to %s", global_step, output_filepath
    )
    parameters = dict(
        itertools.chain.from_iterable(
            [
                (f"spec.{i}.{key}", weights)
                for key, weights in get_state_dict(spec.model).items()
            ]
            for i, spec in enumerate(marketplace)
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


def load_checkpoint(
    marketplace: list[Spec],
    input_filepath: pathlib.Path,
):
    logger.info("Loading checkpoint from %s", input_filepath)
    state = safe_load(input_filepath)

    for i, spec in enumerate(marketplace):
        prefix = f"spec.{i}."
        spec_params = {
            key.removeprefix(prefix): params
            for key, params in state.items()
            if key.startswith(prefix)
        }
        load_state_dict(spec.model, spec_params)

    global_step = state.pop("global_step", None)
    if global_step is not None:
        global_step = global_step.item()
    logger.info(
        "Loaded checkpoint with global_step %s from %s", global_step, input_filepath
    )
