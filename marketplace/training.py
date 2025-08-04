import dataclasses
import functools
import typing

from tinygrad import nn
from tinygrad import Tensor

Model = typing.Callable[[Tensor], Tensor]


@dataclasses.dataclass
class Spec:
    vendors: list[Model]
    upstream_sampling: int = 0


def produce(
    spec: Spec, x: Tensor, paths: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    if paths is None:
        # this is the first spec for taking in the raw input, let's feed data to all of them
        output_data = Tensor.stack(*(vendor(x) for vendor in spec.vendors), dim=0)
        paths = Tensor.arange(len(spec.vendors)).unsqueeze(1)
        return output_data, paths
    if x.size(0) != paths.size(0):
        raise ValueError(
            "Provided input data's first dimension doesn't match with the paths' first dimension"
        )

    upstream_sampling = spec.upstream_sampling
    if upstream_sampling == 0:
        upstream_sampling = x.shape[0]

    input_count = paths.size(0)
    input_indexes = Tensor.stack(
        *(
            Tensor.randperm(input_count)[:upstream_sampling]
            for _ in range(len(spec.vendors))
        ),
        dim=0,
    )
    input_data = x[input_indexes]
    # merge different batches for the same vendor into one. not sure if this is needed, but at least it saves us
    # from calling the model multiple times and making the graph more complex
    merged_batches = input_data.reshape(input_data.shape[0], -1, *input_data.shape[3:])

    output_data = Tensor.stack(
        *(vendor(merged) for vendor, merged in zip(spec.vendors, merged_batches)), dim=0
    )
    # breaking down merged batches back to individual batches
    output_data = output_data.reshape(-1, input_data.shape[2], *output_data.shape[2:])

    prev_paths = paths[input_indexes].flatten(0, 1)
    new_paths = (
        Tensor.arange(len(spec.vendors))
        .unsqueeze(1)
        .repeat(1, upstream_sampling)
        .flatten()
        .unsqueeze(1)
    )
    merged_paths = prev_paths.cat(new_paths, dim=1)

    return output_data, merged_paths


def forward(
    specs: list[Spec], x: Tensor, initial_paths: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    def step(acc: tuple[Tensor, Tensor | None], spec: Spec) -> tuple[Tensor, Tensor]:
        data, paths = acc
        return produce(spec=spec, x=data, paths=paths)

    return functools.reduce(step, specs, (x, initial_paths))


def uniform_between(
    lhs: Tensor,
    rhs: Tensor,
    jitter_scale: Tensor | None = None,
    jitter_offset: Tensor | None = None,
):
    if lhs.shape != rhs.shape:
        raise ValueError("Shape of two tensors should be the same")
    scale = Tensor.uniform(lhs.shape)
    delta = rhs - lhs
    base = lhs
    if jitter_scale is not None:
        base -= delta * jitter_scale
        delta *= 1 + jitter_scale * 2
    if jitter_offset is not None:
        base -= jitter_offset
        delta += jitter_offset * 2
    # Interpolate between two tensor
    return base + delta * scale


def make_offsprings(
    profit_matrix: Tensor, marketplace: list[Spec], offspring_count: int
):
    for vendor_profits, spec in zip(profit_matrix, marketplace):
        reproduce_matrix = (
            vendor_profits.reshape(-1, 1) * vendor_profits.reshape(1, -1)
        ).triu(diagonal=1)
        parent_indexes = reproduce_matrix.flatten().multinomial(
            offspring_count, replacement=True
        )
        lhs_indexes = parent_indexes // vendor_profits.shape[0]
        rhs_indexes = parent_indexes % vendor_profits.shape[0]
        for lhs_idx, rhs_idx in zip(lhs_indexes, rhs_indexes):
            lhs = spec.vendors[lhs_idx.item()]
            rhs = spec.vendors[rhs_idx.item()]
            lhs_params = nn.state.get_state_dict(lhs)
            rhs_params = nn.state.get_state_dict(rhs)
            new_params = {
                uniform_between(
                    lhs=lhs_params[key],
                    rhs=rhs_params[key],
                ).realize()
                for key in lhs_params
            }
            print(new_params)
