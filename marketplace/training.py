import dataclasses
import functools

from tinygrad import nn
from tinygrad import Tensor

from .multi_nn import MultiModelBase


@dataclasses.dataclass
class Spec:
    model: MultiModelBase
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
    model: MultiModelBase,
    x: Tensor,
    paths: Tensor | None = None,
    upstream_sampling: int = 0,
    leading_vendor_index: Tensor | None = None,
    leading_input_index: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Produce various of output for the given model and its vendors with upstream sampling

    :param model: multi-mmodel used to produce output
    :param x: input data from the previous layer
    :param paths: paths for each input data from the previous layer
    :param upstream_sampling: the count of upstream samping from the previous layer. zero means sampling all
    :param leading_vendor_index: the index of leader vendor in the current layer
    :param leading_input_index: the input index of the leading vendor from the previous layer
    :return: (output_data, paths)
    """
    if paths is None:
        # this is the first spec for taking in the raw input, let's feed data to all of them
        output_data = Tensor.stack(
            *(model(Tensor(i), x) for i in range(model.vendor_count)), dim=0
        )
        paths = Tensor.arange(model.vendor_count).unsqueeze(1)
        return output_data, paths
    if x.size(0) != paths.size(0):
        raise ValueError(
            "Provided input data's first dimension doesn't match with the paths' first dimension"
        )

    if upstream_sampling == 0:
        upstream_sampling = x.shape[0]

    input_count = paths.size(0)
    if leading_vendor_index is None:
        # No sticky leader provided it means we are selecting completely randomly from the upstream
        input_indexes = Tensor.stack(
            *(
                Tensor.randperm(input_count)[:upstream_sampling]
                for _ in range(model.vendor_count)
            ),
            dim=0,
        )
    else:
        if leading_vendor_index.ndim != 0:
            raise ValueError("Expected leading_vendor_index to be a scaler tensor")
        if leading_input_index is None:
            raise ValueError(
                "Both leading_vendor_index and leading_input_index needs to be none or non-none"
            )
        if leading_input_index.ndim != 0:
            raise ValueError("Expected leading_input_index to be a scaler tensor")
        # When sticky leader index is provided, it means that we are running in sticky leader mode.
        # TODO: should avoid loop whenever possible to make the compute graph much easier to compile
        input_indexes = Tensor.stack(
            *(
                (i == leading_vendor_index).where(
                    # we are the leading vendor in current layer, let's pick the leading input index and output it
                    # as our first one in the leading vendor's output
                    Tensor.cat(
                        leading_input_index.reshape(1),
                        randperm_skip(upstream_sampling, leading_input_index),
                        dim=0,
                    ),
                    # not leading vendor, let's pick randomly from upstream
                    Tensor.randperm(input_count)[:upstream_sampling],
                )
                for i in Tensor.arange(model.vendor_count)
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

    prev_paths = paths[input_indexes].flatten(0, 1)
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
    leading_path: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    data = x
    paths = initial_paths
    if leading_path is None:
        for spec in specs:
            data, paths = produce(
                model=spec.model,
                x=data,
                paths=paths,
                upstream_sampling=spec.upstream_sampling,
            )
    else:
        leading_input_index = 0
        for spec, leading_vendor_index in zip(specs, leading_path):
            actual_upstream_sampling = spec.upstream_sampling
            if actual_upstream_sampling == 0:
                actual_upstream_sampling = len(data)
            data, paths = produce(
                model=spec.model,
                x=data,
                paths=paths,
                upstream_sampling=spec.upstream_sampling,
                leading_vendor_index=leading_vendor_index,
                leading_input_index=leading_input_index,
            )
            leading_input_index = leading_vendor_index * actual_upstream_sampling
    return data, paths


def forward_with_path(specs: list[Spec], x: Tensor, paths: Tensor) -> Tensor:
    def step(acc: Tensor, spec_idx: tuple[Spec, Tensor]) -> Tensor:
        spec, idx = spec_idx
        return spec.model(idx, acc)

    return functools.reduce(step, zip(specs, paths), x)


def mutate(marketplace: list[Spec], leading_path: Tensor, jitter: Tensor):
    for spec, leading_index in zip(marketplace, leading_path):
        if not spec.evolve:
            continue
        multi_params = nn.state.get_state_dict(spec.model)
        for i in range(spec.model.vendor_count):
            for key, params in multi_params.items():
                if (
                    spec.excluded_param_keys is not None
                    and key in spec.excluded_param_keys
                ):
                    continue
                leading_params = params[leading_index]
                params[i] = (
                    leading_params
                    + (i == leading_index).where(
                        # Do not change the leading vendor
                        Tensor(0),
                        # Copy from the leading vendor and add a bit jitters
                        (
                            Tensor.uniform(
                                *leading_params.shape, low=-jitter, high=jitter
                            )
                        ),
                    )
                ).realize()
