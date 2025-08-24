import copy
import typing

from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import load_state_dict


class StochasticVendor:
    def __init__(self, seed: Tensor):
        self._seed = seed
        self._delta = None
        self._counter = Tensor.zeros(1, dtype=dtypes.uint)

    def __call__(self, model: typing.Callable) -> typing.Callable:
        params = get_state_dict(model)

        if self._deltas is None:
            self._deltas = {
                key: self.make_delta(model, key, param) for key, param in params.items()
            }

        decorated = copy.deepcopy(model)
        load_state_dict(
            decorated,
            state_dict={
                key: param + self._deltas[key] for key, param in params.items()
            },
            verbose=False,
            realize=False,
        )
        return decorated

    def make_delta(self, model: typing.Callable, key: str, params: Tensor) -> Tensor:
        # TODO:
        pass

    def persist(self):
        # TODO: persist delta to model params
        pass
