import copy
import dataclasses
import typing

from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.state import get_state_dict
from tinygrad.nn.state import load_state_dict

from .random import counter_advance_for
from .random import RandomNumberGenerator
from .training import Spec

SEED_MAX = 2**64


@dataclasses.dataclass
class SpecContext:
    seeds: Tensor
    delta: dict[str, Tensor] | None = None
    learning_rates: Tensor | None = None


class CachedDeltaVendor:
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


class DeltaVendor:
    def __init__(self, model: typing.Callable, make_delta: typing.Callable):
        self.model = model
        self.make_delta = make_delta

    def __call__(self, *args, **kwargs):
        vendored_model = copy.deepcopy(self.model)
        model_params = get_state_dict(vendored_model)
        keys = sorted(list(model_params.keys()))
        counter = 0
        vendor_params = {}
        for key in keys:
            params = model_params[key]
            delta = self.make_delta(Tensor(counter, dtype=dtypes.uint), params)
            vendor_params[key] = params + delta
            counter += counter_advance_for(params)

        load_state_dict(
            vendored_model,
            state_dict=vendor_params,
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
        cache_delta: bool = True,
    ):
        self.marketplace = marketplace
        self.learning_rate = learning_rate
        self.make_rng = make_rng
        self.cache_delta = cache_delta

        if seeds is not None:
            market_shape = tuple(spec.vendor_count for spec in marketplace)
            seeds_shape = tuple(len(vendor_seeds) for vendor_seeds in seeds)
            if seeds_shape != market_shape:
                raise ValueError(
                    f"Provided seeds should the same shape {market_shape} as the depth of market but got {seeds_shape}"
                )
        else:
            seeds = [
                Tensor.cat(
                    Tensor.zeros(1, dtype=dtypes.uint64),
                    Tensor.randint(
                        spec.vendor_count - 1, low=1, high=SEED_MAX, dtype=dtypes.uint64
                    ),
                )
                for spec in self.marketplace
            ]

        self.spec_context: list[SpecContext] = [
            SpecContext(
                seeds=seeds[i].contiguous(),
                # allocate memory for delta
                delta=(
                    {
                        key: Tensor.empty(
                            spec.vendor_count, *params.shape, dtype=params.dtype
                        ).contiguous()
                        for key, params in get_state_dict(spec.model).items()
                    }
                    if cache_delta
                    else None
                ),
                learning_rates=self.learning_rate.expand(
                    spec.vendor_count
                ).contiguous(),
            )
            for i, spec in enumerate(self.marketplace)
        ]

        Tensor.realize(
            *(
                # We need to realize all the parameters so that they are buffer instead of compute graph, otherwise the
                # update weights assign operation won't work.
                # ref: https://x.com/fangpenlin/status/1959405151455969607
                [
                    param.assign(param.contiguous())
                    for spec in self.marketplace
                    for param in get_parameters(spec.model)
                ]
                # also realize seeds so that they are buffer
                + [ctx.seeds for ctx in self.spec_context]
            )
        )
        if self.cache_delta:
            # Realize the delta, making them buffers
            Tensor.realize(*self.schedule_delta_update())
            self.vendors = [
                [
                    CachedDeltaVendor(
                        model=spec.model,
                        delta={key: params[i] for key, params in ctx.delta.items()},
                    )
                    for i in range(spec.vendor_count)
                ]
                for spec, ctx in zip(self.marketplace, self.spec_context)
            ]
        else:
            self.vendors = [
                [
                    DeltaVendor(
                        model=spec.model,
                        make_delta=(
                            lambda counter, params, seed=seed, i=i: self.make_delta(
                                seed=seed,
                                counter=counter,
                                lr=self.learning_rate,
                                params=params,
                            )
                        ),
                    )
                    for i, seed in enumerate(ctx.seeds)
                ]
                for spec, ctx in zip(self.marketplace, self.spec_context)
            ]

    def get_seeds(self, path: Tensor) -> Tensor:
        return Tensor.cat(
            *(
                ctx.seeds[index].unsqueeze(0)
                for index, ctx in zip(path, self.spec_context)
            ),
            dim=0,
        )

    def step(self, seeds: Tensor, keep_leader: bool = True):
        Tensor.realize(*self.schedule_step(seeds, keep_leader))

    def schedule_step(self, seeds: Tensor, keep_leader: bool = True) -> list[Tensor]:
        return (
            self.schedule_weight_update(seeds)
            + self.schedule_seeds_update(keep_leader)
            + (self.schedule_delta_update() if self.cache_delta else [])
        )

    def schedule_weight_update(self, seeds: Tensor) -> list[Tensor]:
        weight_updates = []
        for spec, ctx, seed in zip(self.marketplace, self.spec_context, seeds):
            model_params = get_state_dict(spec.model)
            counter = 0

            keys = sorted(list(model_params.keys()))
            for key in keys:
                params = model_params[key]
                weight_updates.append(
                    params.assign(
                        params
                        + self.make_delta(
                            seed=seed,
                            lr=self.learning_rate,
                            counter=Tensor(counter, dtype=dtypes.uint),
                            params=params,
                        )
                    )
                )
                counter += counter_advance_for(params)
        return weight_updates

    def schedule_seeds_update(self, keep_leader: bool = True):
        return [
            ctx.seeds.assign(
                Tensor.cat(
                    Tensor.zeros(1, dtype=dtypes.uint64),
                    Tensor.randint(
                        len(ctx.seeds) - 1, low=1, high=SEED_MAX, dtype=dtypes.uint64
                    ),
                )
                if keep_leader
                else Tensor.randint(
                    *ctx.seeds.shape, low=1, high=SEED_MAX, dtype=dtypes.uint64
                )
            )
            for ctx in self.spec_context
        ]

    def schedule_delta_update(self) -> list[Tensor]:
        if not self.cache_delta:
            raise RuntimeError("Delta cache is not enabled, cannot update delta")
        delta_updates = []
        for ctx in self.spec_context:
            counter = 0
            keys = sorted(list(ctx.delta.keys()))
            for key in keys:
                params = ctx.delta[key]
                updated_params = Tensor.stack(
                    *(
                        self.make_delta(
                            seed=seed,
                            lr=self.learning_rate,
                            counter=Tensor(counter, dtype=dtypes.uint),
                            params=params[i],
                        )
                        for i, seed in enumerate(ctx.seeds)
                    ),
                    dim=0,
                )
                counter += counter_advance_for(params[0])
                delta_updates.append(params.assign(updated_params))
        return delta_updates

    def make_delta(
        self, seed: Tensor, lr: Tensor, counter: Tensor, params: Tensor
    ) -> Tensor:
        return (seed != 0).where(
            self.make_rng(seed=seed, counter=counter).uniform_like(
                params,
                low=-lr,
                high=lr,
            ),
            Tensor.zeros_like(params),
        )
