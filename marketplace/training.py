import dataclasses

from tinygrad import dtypes
from tinygrad import nn
from tinygrad import Tensor

from .delta_nn import DeltaModelBase
from .multi_nn import MultiModelBase
from .random import RandomNumberGenerator

SEED_MAX = 2**64


@dataclasses.dataclass
class Spec:
    model: DeltaModelBase
    vendor_count: int
    upstream_sampling: int = 0
    evolve: bool = True
    excluded_param_keys: frozenset[str] | None = None


def randperm_skip(size: int, skip_index: Tensor) -> Tensor:
    """The same as randperm but skip given index `skip_index`

    :param size: size of randperm
    :param skip_index: index to skip
    :return: A tensor of random permutation from 0 to N-1 without the skip_index in it
    """
    indexes = Tensor.arange(size - 1)
    offsets = (indexes >= skip_index).cast(indexes.dtype)
    shifted_indexes = indexes + offsets
    perm = Tensor.randperm(size - 1)
    return shifted_indexes[perm]


def produce(
    model: DeltaModelBase,
    vendor_count: int,
    x: Tensor,
    seeds: Tensor | None = None,
    upstream_sampling: int = 0,
) -> tuple[Tensor, Tensor]:
    """Produce various of output for the given model and its vendors with upstream sampling

    :param model: multi-mmodel used to produce output
    :param x: input data from the previous layer
    :param seeds: paths for each input data from the previous layer
    :param upstream_sampling: the count of upstream samping from the previous layer. zero means sampling all
    :return: (output_data, paths)
    """
    if seeds is None:
        seeds = Tensor.randint(vendor_count, low=0, high=SEED_MAX, dtype=dtypes.uint64)
        counters = Tensor.zeros(vendor_count, dtype=dtypes.int)

        # this is the first spec for taking in the raw input, let's feed data to all of them
        # TODO: use RANGIFY feature when it's ready to make JIT's job much easier
        output_data = Tensor.stack(
            *(
                model(RandomNumberGenerator(seed=seed, counter=counter), x)
                for seed, counter in zip(seeds, counters)
            ),
            dim=0,
        )
        return output_data, seeds.unsqueeze(1)
    if x.size(0) != seeds.size(0):
        raise ValueError(
            "Provided input data's first dimension doesn't match with the paths' first dimension"
        )

    if upstream_sampling == 0:
        # when upstream sampling is zero, it means we sample the full input
        upstream_sampling = x.shape[0]
        input_indexes = Tensor.arange(x.shape[0]).repeat(model.vendor_count, 1)
    else:
        input_count = seeds.size(0)
        input_indexes = Tensor.stack(
            *(
                Tensor.randperm(input_count)[:upstream_sampling]
                for _ in range(model.vendor_count)
            ),
            dim=0,
        )

    input_data = x[input_indexes]
    # merge different batches for the same vendor into one. not sure if this is needed, but at least it saves us
    # from calling the model multiple times and making the graph more complex
    merged_batches = input_data.reshape(input_data.shape[0], -1, *input_data.shape[3:])

    output_data = Tensor.stack(
        *(
            model(i, merged)
            for i, merged in zip(range(model.vendor_count), merged_batches)
        ),
        dim=0,
    )
    # breaking down merged batches back to individual batches
    output_data = output_data.reshape(-1, input_data.shape[2], *output_data.shape[2:])

    prev_paths = seeds[input_indexes].flatten(0, 1)
    new_paths = (
        Tensor.arange(model.vendor_count)
        .unsqueeze(1)
        .repeat(1, upstream_sampling)
        .flatten()
        .unsqueeze(1)
    )
    merged_paths = prev_paths.cat(new_paths, dim=1)

    return output_data, merged_paths


def forward(
    specs: list[Spec],
    x: Tensor,
    initial_paths: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    data = x
    paths = initial_paths
    for spec in specs:
        data, paths = produce(
            model=spec.model,
            x=data,
            seeds=paths,
            upstream_sampling=spec.upstream_sampling,
        )
    return data, paths


def forward_with_path(specs: list[Spec], x: Tensor, paths: Tensor) -> Tensor:
    data = x
    for spec, index in zip(specs, paths):
        data = spec.model(index, data)
    return data


def traverse(model: MultiModelBase, path: list[str]) -> MultiModelBase:
    current = model
    while path:
        key = path.pop(0)
        if isinstance(current, (list, tuple)):
            current = current[int(key)]
        elif isinstance(current, dict):
            current = current[key]
        else:
            current = getattr(current, key)
    return current


def mutate(marketplace: list[Spec], leading_path: Tensor, jitter: Tensor):
    for spec, leading_index in zip(marketplace, leading_path):
        if not spec.evolve:
            continue
        multi_params = nn.state.get_state_dict(spec.model)
        for key, params in multi_params.items():
            if spec.excluded_param_keys is not None and key in spec.excluded_param_keys:
                continue
            leading_params = params[leading_index]
            path = key.split(".")
            owner_model = traverse(spec.model, path[:-1])
            copy_only_params = ()
            if hasattr(owner_model, "copy_only_params"):
                copy_only_params = owner_model.copy_only_params
            attr_name = path[-1]
            if attr_name in copy_only_params:
                params.assign(
                    leading_params.repeat(
                        spec.model.vendor_count, *((1,) * leading_params.ndim)
                    )
                ).realize()
                continue
            delta = Tensor.uniform(*params.shape, low=-jitter, high=jitter)
            # TODO: by generating a big block of random number and masking the part we don't want to change with
            #       zeros, while it runs faster overall, but we waste time generating random numbers not really used...
            delta[leading_index].assign(Tensor.zeros(*leading_params.shape))
            params.assign(
                leading_params.repeat(
                    spec.model.vendor_count, *((1,) * leading_params.ndim)
                )
                + delta
            ).realize()
