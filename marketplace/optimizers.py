import copy
import typing
from collections import OrderedDict

from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import load_state_dict

from .random import rand
from .training import Spec

SEED_MAX = 2**64


class StochasticVendor:
    def __init__(self, seed: Tensor, learning_rate: Tensor):
        self.seed = seed
        self.learning_rate = learning_rate
        self.delta = None
        self.counter = Tensor(0, dtype=dtypes.uint).realize()

    def __call__(self, model: typing.Callable) -> typing.Callable:
        params = get_state_dict(model)

        if self.delta is None:
            self.delta = OrderedDict(
                [
                    (key, self.make_delta(param).realize())
                    for key, param in params.items()
                ]
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
        # TODO: take RNG from the external to make it possible to use different RNG algorithm?
        # TODO: deal with zero seed
        return (
            (high - low) * rand(*params.shape, seed=self.seed, counter=self.counter)
        ).cast(params.dtype or dtypes.default_float) + low

    def schedule_delta_update(self) -> list[Tensor]:
        if self.delta is None:
            return []
        return [self.counter.assign(Tensor(0, dtype=dtypes.uint))] + [
            param.assign(self.make_delta(param)) for param in self.delta.values()
        ]

    def schedule_weight_update(self, model: typing.Callable) -> list[Tensor]:
        params = get_state_dict(model)
        return load_state_dict(
            model,
            state_dict={key: param + self.delta[key] for key, param in params.items()},
            verbose=False,
            realize=False,
        )


class StochasticOptimizer:
    def __init__(self, marketplace: list[Spec], learning_rate: Tensor):
        self.marketplace = marketplace
        self.learning_rate = learning_rate

        self.seeds = [
            Tensor.randint(spec.vendor_count, low=1, high=SEED_MAX, dtype=dtypes.uint64)
            for spec in self.marketplace
        ]
        Tensor.realize(
            *(
                # We need to realize all the parameters so that they are buffer instead of compute graph, otherwise the assign
                # operation won't work.
                # ref: https://x.com/fangpenlin/status/1959405151455969607
                [
                    param
                    for spec in self.marketplace
                    for param in get_parameters(spec.model)
                ]
                # also realize seeds so that they are buffer
                + self.seeds
            )
        )
        self.vendors = [
            [
                StochasticVendor(seed=seed, learning_rate=self.learning_rate)
                for seed in vendor_seeds
            ]
            for vendor_seeds in self.seeds
        ]

    def step(self, seeds: Tensor):
        Tensor.realize(*self.schedule_step(seeds))

    def schedule_step(self, seeds: Tensor) -> list[Tensor]:
        return [
            param
            for spec, seed in zip(self.marketplace, seeds)
            for param in spec.model.update(self.make_rng(seed)).values()
        ]
