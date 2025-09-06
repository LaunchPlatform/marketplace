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
    delta: dict[str, Tensor]
    learning_rate: Tensor
    learning_rate_scales: Tensor | None = None


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


class Optimizer:
    def __init__(
        self,
        marketplace: list[Spec],
        learning_rate: Tensor,
        # learning_rate_scale_range: Tensor | None = None,
        meta_learning_rate: Tensor | None = None,
        seeds: list[Tensor] | None = None,
        probe_scale: Tensor | None = None,
        make_rng: typing.Type[RandomNumberGenerator] = RandomNumberGenerator,
    ):
        self.marketplace = marketplace
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.make_rng = make_rng
        self.probe_scale = probe_scale

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
                delta={
                    key: Tensor.empty(
                        spec.vendor_count, *params.shape, dtype=params.dtype
                    ).contiguous()
                    for key, params in get_state_dict(spec.model).items()
                },
                learning_rate=(
                    self.learning_rate.clone().contiguous()
                    if self.meta_learning_rate is not None
                    else self.learning_rate
                ),
                learning_rate_scales=(
                    Tensor.zeros(spec.vendor_count).contiguous()
                    if self.meta_learning_rate is not None
                    else None
                ),
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

    def get_seeds(self, path: Tensor) -> Tensor:
        return Tensor.cat(
            *(
                ctx.seeds[index].unsqueeze(0)
                for index, ctx in zip(path, self.spec_context)
            ),
            dim=0,
        )

    def get_learning_rates(self, path: Tensor) -> Tensor:
        return Tensor.stack(
            *(
                ctx.learning_rate * (1 + ctx.learning_rate_scales[index])
                for index, ctx in zip(path, self.spec_context)
            ),
            dim=0,
        )

    def step(
        self,
        seeds: Tensor,
        learning_rates: Tensor | None = None,
        keep_leader: bool = True,
    ):
        Tensor.realize(
            *self.schedule_step(
                seeds, learning_rates=learning_rates, keep_leader=keep_leader
            )
        )

    def schedule_step(
        self,
        seeds: Tensor,
        learning_rates: Tensor | None = None,
        keep_leader: bool = True,
    ) -> list[Tensor]:
        return (
            self.schedule_weight_update(seeds, learning_rates=learning_rates)
            + self.schedule_seeds_update(keep_leader)
            + self.schedule_delta_update()
        )

    def schedule_weight_update(
        self,
        direction_delta: list[dict[str, Tensor]],
        learning_rates: Tensor | None = None,
    ) -> list[Tensor]:
        weight_updates = []
        if learning_rates is None:
            learning_rates = self.learning_rate.expand(len(self.marketplace))
        for spec, ctx, delta, lr in zip(
            self.marketplace, self.spec_context, direction_delta, learning_rates
        ):
            model_params = get_state_dict(spec.model)
            keys = sorted(list(model_params.keys()))
            effective_lr = ctx.learning_rate
            if self.meta_learning_rate is not None:
                weight_updates.append(ctx.learning_rate.assign(lr))
                effective_lr = lr
            for key in keys:
                params = model_params[key]
                weight_updates.append(params.assign(params + delta[key] * effective_lr))
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
                            lr=(
                                ctx.learning_rate
                                if self.probe_scale is None
                                else (ctx.learning_rate * self.probe_scale)
                            ),
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

    def schedule_lr_scale_update(self, direction_vectors: Tensor) -> list[Tensor]:
        if self.meta_learning_rate is None:
            raise ValueError("LR scale not set")
        lr_updates = []
        for ctx, vector in zip(self.spec_context, direction_vectors):
            # We use the final counter (after all params) for generating the lr delta.
            # TODO: extract this part?
            final_counter = 0
            keys = sorted(list(ctx.delta.keys()))
            for key in keys:
                params = ctx.delta[key]
                final_counter += counter_advance_for(params[0])
            # Generate different LR to try out
            lr_updates.append(
                ctx.learning_rate_scales.assign(
                    Tensor.stack(
                        *(
                            (
                                self.make_delta(
                                    seed=seed,
                                    counter=Tensor(final_counter, dtype=dtypes.uint),
                                    lr=self.meta_learning_rate,
                                    params=lr_scale,
                                )
                                if i != 0
                                # we always keep the original lr in the combinations, in case we cannot find any
                                # improvement from scale, at least we are not making regression
                                else Tensor.zeros_like(lr_scale)
                            )
                            for i, (lr_scale, seed) in enumerate(
                                zip(ctx.learning_rate_scales, ctx.seeds)
                            )
                        ),
                        dim=0,
                    )
                )
            )
            for key in keys:
                params = ctx.delta[key]
                updated_params = Tensor.stack(
                    *(
                        vector[key] * (ctx.learning_rate * (1 + lr_scale))
                        for i, lr_scale in enumerate(ctx.learning_rate_scales)
                    ),
                    dim=0,
                )
                lr_updates.append(params.assign(updated_params))
        return lr_updates

    def compute_direction_vectors(
        self, loss: Tensor, paths: Tensor
    ) -> list[dict[str, Tensor]]:
        std, mean = loss.std_mean()
        std_loss = -((loss - mean) / std)
        reconciled_deltas = []
        vector_square_sum = []
        for i, (spec, ctx) in enumerate(zip(self.marketplace, self.spec_context)):
            model_params = get_state_dict(spec.model)
            keys = sorted(list(model_params.keys()))
            counter = 0
            indexes = paths[:, i]
            reconciled_delta = {}
            for key in keys:
                reconciled_delta[key] = (
                    # Take all the delta and multiply their corresponding normalized loss, so that we can "reward" each
                    # parameters in delta accordingly to compose a overall better direction.
                    ctx.delta[key][indexes]
                    * std_loss.reshape(
                        len(std_loss), *((1,) * len(model_params[key].shape))
                    )
                ).sum(axis=0)
                counter += counter_advance_for(model_params[key])
            reconciled_deltas.append(reconciled_delta)
            # We treat all the parameters delta in this spec as a vector
            combined_vector = Tensor.cat(
                *[delta.flatten() for delta in reconciled_delta.values()]
            )
            # add up the vector's element^2
            vector_square_sum.append(combined_vector.square().sum().unsqueeze(0))
        vector_len = Tensor.cat(*vector_square_sum).sum().sqrt()
        return [
            # make them a unit vector
            {key: delta / vector_len for key, delta in reconciled_delta.items()}
            for reconciled_delta in reconciled_deltas
        ]

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
