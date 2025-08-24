import copy
import typing
from collections import OrderedDict

from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import load_state_dict

from .random import rand


class StochasticVendor:
    def __init__(self, seed: Tensor, learning_rate: Tensor):
        self.seed = seed
        self.learning_rate = learning_rate
        self.delta = None
        self.counter = Tensor.zeros(1, dtype=dtypes.uint)

    def __call__(self, model: typing.Callable) -> typing.Callable:
        params = get_state_dict(model)

        if self.delta is None:
            self.delta = OrderedDict(
                [(key, self.make_delta(param)) for key, param in params.items()]
            )

        decorated = copy.deepcopy(model)
        load_state_dict(
            decorated,
            state_dict={key: param + self.delta[key] for key, param in params.items()},
            verbose=False,
            realize=False,
        )
        return decorated

    def make_delta(self, params: Tensor) -> Tensor:
        high = self.learning_rate
        low = -self.learning_rate
        return (
            (high - low) * rand(*params.shape, seed=self.seed, counter=self.counter)
        ).cast(params.dtype or dtypes.default_float) + low

    def schedule_seed_update(self, seed: Tensor) -> list[Tensor]:
        updates = [self.seed.assign(seed)]
        if self.delta is None:
            return updates
        for param in self.delta.values():
            updates.append(param.assign(self.make_delta(param)))
        return updates

    def persist(self):
        # TODO: persist delta to model params
        pass
