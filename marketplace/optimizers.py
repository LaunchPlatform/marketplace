import copy
import typing

from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import load_state_dict

from .random import rand
from .training import Spec

SEED_MAX = 2**64


class DeltaVendor:
    def __init__(self, model: typing.Callable, delta: dict[str, Tensor]):
        self.model = model
        self.delta = delta

        params = get_state_dict(model)
        self.vendored_model = copy.deepcopy(model)
        load_state_dict(
            self.vendored_model,
            state_dict={key: param + self.delta[key] for key, param in params.items()},
            verbose=False,
            realize=False,
        )

    def __call__(self, *args, **kwargs):
        return self.vendored_model(*args, **kwargs)


class StochasticOptimizer:
    def __init__(self, marketplace: list[Spec], learning_rate: Tensor):
        self.marketplace = marketplace
        self.learning_rate = learning_rate

        self.seeds = [
            Tensor.randint(spec.vendor_count, low=1, high=SEED_MAX, dtype=dtypes.uint64)
            for spec in self.marketplace
        ]
        self.counters = [Tensor.zeros(spec.vendor_count) for spec in self.marketplace]
        Tensor.realize(
            *(
                # We need to realize all the parameters so that they are buffer instead of compute graph, otherwise the
                # update weights assign operation won't work.
                # ref: https://x.com/fangpenlin/status/1959405151455969607
                [
                    param
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
                key: Tensor.empty(*params.shape, dtype=params.dtype)
                .expand(spec.vendor_count, *params.shape)
                .contiguous()
                for key, params in get_state_dict(spec.model).items()
            }
            for spec, vendor_seeds, vendor_counters in zip(
                self.marketplace, self.seeds, self.counters
            )
        ]
        # Realize the delta, making them buffers
        Tensor.realize(*self.schedule_delta_update())
        self.vendors = [
            [
                DeltaVendor(
                    model=spec.model,
                    delta={key: params[i] for key, params in vendor_deltas.items()},
                )
                for i in range(spec.vendor_count)
            ]
            for spec, vendor_deltas in zip(self.marketplace, self.delta)
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
        return [
            param.assign(param + vendor_deltas[key][index])
            for spec, vendor_deltas, index in zip(self.marketplace, self.delta, path)
            for key, param in get_state_dict(spec.model).items()
        ]

    def schedule_seeds_update(self, keep_leader: bool = True):
        return [
            vendor_seeds.assign(
                Tensor.cat(
                    Tensor.zeros(1, dtype=dtypes.uint64),
                    Tensor.randint(
                        len(vendor_seeds) - 1, low=1, high=SEED_MAX, dtype=dtypes.uint64
                    ),
                )
                if keep_leader
                else Tensor.randint(
                    *vendor_seeds.shape, low=1, high=SEED_MAX, dtype=dtypes.uint64
                )
            )
            for vendor_seeds in self.seeds
        ]

    def schedule_delta_update(self) -> list[Tensor]:
        return (
            # reset the rng counters
            [
                vendor_counters.assign(Tensor.zeros_like(vendor_counters))
                for vendor_counters in self.counters
            ]
            # generate new delta based on the current seed and lr
            + [
                params.assign(self.make_delta(seed, counter, params))
                for vendor_deltas, vendor_seeds, vendor_counters in zip(
                    self.delta, self.seeds, self.counters
                )
                for params, seed, counter in zip(
                    vendor_deltas.values(), vendor_seeds, vendor_counters
                )
            ]
        )

    def make_delta(self, seed: Tensor, counter: Tensor, params: Tensor) -> Tensor:
        high = self.learning_rate
        low = -self.learning_rate
        # TODO: take RNG from the external to make it possible to use different RNG algorithm?
        uniform = ((high - low) * rand(*params.shape, seed=seed, counter=counter)).cast(
            params.dtype or dtypes.default_float
        ) + low
        return (seed != 0).where(
            uniform,
            Tensor.zeros_like(params),
        )
