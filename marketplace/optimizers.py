import copy
import typing

from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import load_state_dict

from .random import RandomNumberGenerator
from .training import Spec

SEED_MAX = 2**64


class DeltaVendor:
    def __init__(self, model: typing.Callable, delta: dict[str, Tensor]):
        self.vendored_model = copy.deepcopy(model)
        load_state_dict(
            self.vendored_model,
            state_dict={
                key: param + delta[key]
                for key, param in get_state_dict(self.vendored_model).items()
            },
            verbose=False,
            realize=False,
        )

    def __call__(self, *args, **kwargs):
        return self.vendored_model(*args, **kwargs)


class StochasticOptimizer:
    def __init__(
        self,
        marketplace: list[Spec],
        learning_rate: Tensor,
        seeds: list[Tensor] | None = None,
        make_rng: typing.Type[RandomNumberGenerator] = RandomNumberGenerator,
    ):
        self.marketplace = marketplace
        self.learning_rate = learning_rate
        self.make_rng = make_rng

        if seeds is not None:
            market_shape = tuple(spec.vendor_count for spec in marketplace)
            seeds_shape = tuple(len(vendor_seeds) for vendor_seeds in seeds)
            if seeds_shape != market_shape:
                raise ValueError(
                    f"Provided seeds should the same shape {market_shape} as the depth of market but got {seeds_shape}"
                )
            self.seeds = seeds
        else:
            self.seeds = [
                Tensor.cat(
                    Tensor.zeros(1, dtype=dtypes.uint64),
                    Tensor.randint(
                        spec.vendor_count - 1, low=1, high=SEED_MAX, dtype=dtypes.uint64
                    ),
                )
                for spec in self.marketplace
            ]
        self.counters = [
            Tensor.zeros(spec.vendor_count, dtype=dtypes.uint).contiguous()
            for spec in self.marketplace
        ]
        Tensor.realize(
            *(
                # We need to realize all the parameters so that they are buffer instead of compute graph, otherwise the
                # update weights assign operation won't work.
                # ref: https://x.com/fangpenlin/status/1959405151455969607
                [
                    param.contiguous()
                    for spec in self.marketplace
                    for param in get_parameters(spec.model)
                ]
                # also realize seeds and counters so that they are buffer
                + self.seeds
                + self.counters
            )
        )

        # Allocate memory for parameter delta
        self.delta = [
            {
                key: Tensor.empty(
                    spec.vendor_count, *params.shape, dtype=params.dtype
                ).contiguous()
                for key, params in get_state_dict(spec.model).items()
            }
            for spec in self.marketplace
        ]
        # Realize the delta, making them buffers
        Tensor.realize(*self.schedule_delta_update())
        self.vendors = [
            [
                DeltaVendor(
                    model=spec.model,
                    delta={key: params[i] for key, params in deltas.items()},
                )
                for i in range(spec.vendor_count)
            ]
            for spec, deltas in zip(self.marketplace, self.delta)
        ]

    def step(self, path: Tensor, keep_leader: bool = True):
        Tensor.realize(*self.schedule_step(path, keep_leader))

    def schedule_step(self, path: Tensor, keep_leader: bool = True) -> list[Tensor]:
        return (
            self.schedule_weight_update(path)
            + self.schedule_seeds_update(keep_leader)
            + self.schedule_delta_update()
        )

    def schedule_weight_update(self, path: Tensor) -> list[Tensor]:
        weight_updates = []
        for spec, deltas, index in zip(self.marketplace, self.delta, path):
            for key, param in get_state_dict(spec.model).items():
                weight_updates.append(param.assign(param + deltas[key][index]))
        return weight_updates

    def schedule_seeds_update(self, keep_leader: bool = True):
        return [
            seeds.assign(
                Tensor.cat(
                    Tensor.zeros(1, dtype=dtypes.uint64),
                    Tensor.randint(
                        len(seeds) - 1, low=1, high=SEED_MAX, dtype=dtypes.uint64
                    ),
                )
                if keep_leader
                else Tensor.randint(
                    *seeds.shape, low=1, high=SEED_MAX, dtype=dtypes.uint64
                )
            )
            for seeds in self.seeds
        ]

    def schedule_delta_update(self) -> list[Tensor]:
        counter_resets = [
            counters.assign(Tensor.zeros_like(counters)) for counters in self.counters
        ]
        delta_updates = []
        for deltas, seeds, counters in zip(self.delta, self.seeds, self.counters):
            for params_delta in deltas.values():
                updated_params = Tensor.stack(
                    *(
                        self.make_delta(
                            seed=seed, counter=counter, params=params_delta[i]
                        )
                        for i, (seed, counter) in enumerate(zip(seeds, counters))
                    ),
                    dim=0,
                )
                delta_updates.append(params_delta.assign(updated_params))
        return counter_resets + delta_updates

    def make_delta(self, seed: Tensor, counter: Tensor, params: Tensor) -> Tensor:
        return (seed != 0).where(
            self.make_rng(seed=seed, counter=counter).uniform_like(
                params, low=-self.learning_rate, high=self.learning_rate
            ),
            Tensor.zeros_like(params),
        )
