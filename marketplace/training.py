import dataclasses
import typing

from tinygrad import Tensor
from tinygrad.nn.state import get_parameters

from .nn import ModelBase
from .random import RandomNumberGenerator

SEED_MAX = 2**64
RandomNumberGeneratorFactory = typing.Callable[[Tensor], RandomNumberGenerator]


@dataclasses.dataclass
class Spec:
    model: ModelBase
    vendor_count: int
    upstream_sampling: int = 0
    evolve: bool = True


class Optimizer:
    def __init__(self, marketplace: list[Spec], make_rng: RandomNumberGeneratorFactory):
        self.marketplace = marketplace
        self.make_rng = make_rng
        # We need to realize all the parameters so that they are buffer instead of compute graph, otherwise the assign
        # operation won't work.
        # ref: https://x.com/fangpenlin/status/1959405151455969607
        Tensor.realize(
            *(
                param
                for spec in self.marketplace
                for param in get_parameters(spec.model)
            )
        )

    def step(self, seeds: Tensor):
        Tensor.realize(*self.schedule_step(seeds))

    def schedule_step(self, seeds: Tensor) -> list[Tensor]:
        return [
            param
            for spec, seed in zip(self.marketplace, seeds)
            for param in spec.model.update(self.make_rng(seed)).values()
        ]


class VendorOptimizer:
    def __call__(self, *args, **kwargs):
        pass


def produce(
    spec: Spec,
    x: Tensor,
    optimizers: list,
    seeds: Tensor,
    acc_seeds: Tensor | None = None,
    upstream_sampling: int = 0,
) -> tuple[Tensor, Tensor]:
    """Produce various of output for the given model and its vendors with upstream sampling

    :param spec: spec of marketplace
    :param x: input data from the previous layer
    :param optimizers: optimizers for each vendor
    :param seeds: seeds for each vendor
    :param acc_seeds: accumulated seeds so far from the previous layers
    :param upstream_sampling: the count of upstream samping from the previous layer. zero means sampling all
    :return: (output_data, seeds)
    """
    if acc_seeds is None:
        # this is the first spec for taking in the raw input, let's feed data to all of them
        # TODO: use RANGIFY feature when it's ready to make JIT's job much easier
        output_data = Tensor.stack(
            *(decorator(spec.model)(x) for decorator, _ in optimizers),
            dim=0,
        )
        return output_data, seeds
    if x.size(0) != acc_seeds.size(0):
        raise ValueError(
            "Provided input data's first dimension doesn't match with the seeds' first dimension"
        )

    if upstream_sampling == 0:
        # when upstream sampling is zero, it means we sample the full input
        upstream_sampling = x.shape[0]
        input_indexes = Tensor.arange(x.shape[0]).expand(spec.vendor_count, -1)
    else:
        input_count = acc_seeds.size(0)
        # TODO: use RANGIFY?
        input_indexes = Tensor.stack(
            *(
                Tensor.randperm(input_count)[:upstream_sampling]
                for _ in range(spec.vendor_count)
            ),
            dim=0,
        )

    input_data = x[input_indexes]
    # merge different batches for the same vendor into one. not sure if this is needed, but at least it saves us
    # from calling the model multiple times and making the graph more complex
    merged_batches = input_data.reshape(input_data.shape[0], -1, *input_data.shape[3:])

    output_data = Tensor.stack(
        *(
            optimizer(spec.model)(merged)
            for optimizer, merged in zip(optimizers, merged_batches)
        ),
        dim=0,
    )
    # breaking down merged batches back to individual batches
    output_data = output_data.reshape(-1, input_data.shape[2], *output_data.shape[2:])

    prev_seeds = acc_seeds[input_indexes].flatten(0, 1)
    current_seeds = (
        seeds.unsqueeze(1).repeat(1, upstream_sampling).flatten().unsqueeze(1)
    )
    merged_seeds = prev_seeds.cat(current_seeds, dim=1)
    return output_data, merged_seeds


def forward(
    marketplace: list[Spec],
    x: Tensor,
    optimizer: Optimizer,
    initial_seeds: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    data = x
    acc_seeds = initial_seeds
    for spec, seeds in zip(marketplace, vendor_seeds):
        data, acc_seeds = produce(
            make_rng=make_rng,
            spec=spec,
            x=data,
            seeds=seeds,
            acc_seeds=acc_seeds,
            upstream_sampling=spec.upstream_sampling,
        )
    return data, acc_seeds


def straight_forward(specs: list[Spec], x: Tensor) -> Tensor:
    data = x
    for spec in specs:
        data = spec.model(data)
    return data
