import copy
import typing
from collections import OrderedDict

from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import load_state_dict

from .random import counter_advance_for
from .random import RandomNumberGenerator
from .training import Spec

SEED_MAX = 2**64


class DeltaVendor:
    def __init__(self, model: typing.Callable, delta: dict[str, Tensor]):
        self.model = model
        self.delta = delta

    def __call__(self, *args, **kwargs):
        vendored_model = copy.deepcopy(self.model)
        load_state_dict(
            vendored_model,
            state_dict={
                key: param + self.delta[key]
                for key, param in get_state_dict(vendored_model).items()
            },
            verbose=False,
            realize=False,
        )
        return vendored_model(*args, **kwargs)


class Optimizer:
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
                # also realize seeds so that they are buffer
                + self.seeds
            )
        )

        # Allocate memory for parameter delta
        self.delta = [
            # Use order dict to keep the order, so that we can use the same order to generate random numbers in the
            # same sequence. Otherwise, if the sequence is wrong, we will end up with the wrong random numbers
            OrderedDict(
                [
                    (
                        key,
                        Tensor.empty(
                            spec.vendor_count, *params.shape, dtype=params.dtype
                        ).contiguous(),
                    )
                    for key, params in get_state_dict(spec.model).items()
                ]
            )
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

    def get_seeds(self, path: Tensor) -> Tensor:
        return Tensor.cat(
            *(seeds[index].unsqueeze(0) for index, seeds in zip(path, self.seeds))
        )

    def step(self, best_seeds: Tensor, keep_leader: bool = True):
        Tensor.realize(*self.schedule_step(best_seeds, keep_leader))

    def schedule_step(
        self, best_seeds: Tensor, keep_leader: bool = True
    ) -> list[Tensor]:
        return (
            self.schedule_weight_update(best_seeds)
            + self.schedule_seeds_update(keep_leader)
            + self.schedule_delta_update()
        )

    def schedule_weight_update(self, best_seeds: Tensor) -> list[Tensor]:
        weight_updates = []
        for spec, deltas, seed in zip(self.marketplace, self.delta, best_seeds):
            model_params = get_state_dict(spec.model)
            counter = Tensor.zeros(dtype=dtypes.uint)
            # Notice: we use deltas because it's an ordered dict, we want to have the same order for making random
            #         numbers
            for key in deltas:
                params = model_params[key]
                weight_updates.append(
                    params.assign(
                        params
                        + self.make_delta(seed=seed, counter=counter, params=params)
                    )
                )
                counter += counter_advance_for(params)
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
        delta_updates = []
        for deltas, seeds in zip(self.delta, self.seeds):
            counter = Tensor.zeros(dtype=dtypes.uint)
            for params_delta in deltas.values():
                updated_params = Tensor.stack(
                    *(
                        self.make_delta(
                            seed=seed, counter=counter, params=params_delta[i]
                        )
                        for i, seed in enumerate(seeds)
                    ),
                    dim=0,
                )
                counter += counter_advance_for(params_delta[0])
                delta_updates.append(params_delta.assign(updated_params))
        return delta_updates

    def make_delta(self, seed: Tensor, counter: Tensor, params: Tensor) -> Tensor:
        return (seed != 0).where(
            self.make_rng(seed=seed, counter=counter).uniform_like(
                params, low=-self.learning_rate, high=self.learning_rate
            ),
            Tensor.zeros_like(params),
        )
