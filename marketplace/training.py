import dataclasses
import typing

from tinygrad import Tensor


@dataclasses.dataclass
class Spec:
    model: typing.Callable
    vendor_count: int
    upstream_sampling: int = 0


def produce(
    spec: Spec,
    x: Tensor,
    vendors: list[typing.Callable],
    paths: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Produce various of output for the given model and its vendors with upstream sampling

    :param spec: spec of marketplace
    :param x: raw input data or intermediate products from the previous layer
    :param vendors: vendors for decorating a model
    :param paths: accumulated paths so far from the previous layers
    :return: (output_data, paths)
    """
    if paths is None:
        # this is the first spec for taking in the raw input, let's feed data to all of them
        # TODO: use RANGIFY feature when it's ready to make JIT's job much easier
        output_data = Tensor.stack(
            *(vendor(x) for vendor in vendors),
            dim=0,
        )
        paths = Tensor.arange(len(vendors)).unsqueeze(1)
        return output_data, paths
    if x.size(0) != paths.size(0):
        raise ValueError(
            "Provided input data's first dimension doesn't match with the paths' first dimension"
        )

    if spec.upstream_sampling == 0:
        # when upstream sampling is zero, it means we sample the full input
        upstream_sampling = x.shape[0]
        input_indexes = Tensor.arange(x.shape[0]).expand(spec.vendor_count, -1)
    else:
        upstream_sampling = spec.upstream_sampling
        input_count = paths.size(0)
        # TODO: use RANGIFY?
        input_indexes = Tensor.stack(
            *(
                Tensor.randperm(input_count)[:upstream_sampling]
                for _ in range(spec.vendor_count)
            ),
            dim=0,
        )

    input_data = x[input_indexes]
    # merge different batches for the same vendor into one.
    merged_batches = input_data.reshape(input_data.shape[0], -1, *input_data.shape[3:])

    output_data = Tensor.stack(
        *(vendor(merged) for vendor, merged in zip(vendors, merged_batches)),
        dim=0,
    )
    # breaking down merged batches back to individual batches
    output_data = output_data.reshape(-1, input_data.shape[2], *output_data.shape[2:])

    prev_paths = paths[input_indexes].flatten(0, 1)
    current_paths = (
        Tensor.arange(len(vendors))
        .unsqueeze(1)
        .repeat(1, upstream_sampling)
        .flatten()
        .unsqueeze(1)
    )
    merged_paths = prev_paths.cat(current_paths, dim=1)
    return output_data, merged_paths


def forward(
    marketplace: list[Spec],
    x: Tensor,
    vendors: list[list[typing.Callable]],
    initial_paths: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    data = x
    acc_paths = initial_paths
    for spec, spec_vendors in zip(marketplace, vendors):
        data, acc_paths = produce(
            spec=spec,
            x=data,
            vendors=spec_vendors,
            paths=acc_paths,
        )
    return data, acc_paths


def straight_forward(specs: list[Spec], x: Tensor) -> Tensor:
    data = x
    for spec in specs:
        data = spec.model(data)
    return data
