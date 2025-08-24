import copy
import typing

from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import load_state_dict


class StochasticVendor:
    def __init__(self, seed: Tensor):
        self._seed = seed

    def __call__(self, model: typing.Callable) -> typing.Callable:
        def callee(*args, **kwargs):
            params = get_state_dict(model)

            # TODO: if deltas not defined
            self._deltas = {key: delta_like(param) for key, param in params.items()}

            decorated = copy.deepcopy(model)
            load_state_dict(
                decorated,
                state_dict={
                    key: param + self._deltas[key] for key, param in params.items()
                },
                verbose=False,
                realize=False,
            )

            return decorated(*args, **kwargs)

        return callee

    def persist(self):
        # TODO: persist delta to model params
        pass
