import dataclasses
import typing

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

    input_count = paths.size(0)
    input_indexes = Tensor.stack(
        *(
            Tensor.randperm(input_count)[: spec.upstream_sampling]
            for _ in range(len(spec.vendors))
        ),
        dim=0,
    )
    input_data = x[input_indexes]
    # merge different batches for the same vendor into one. not sure if this is needed, but at least it saves us
    # from calling the model multiple times and making the graph more complex
    merged_batches = input_data.reshape(input_data.shape[0], -1, *input_data.shape[3:])

    output_data = Tensor.stack(
        *(vendor(merged) for merged, vendor in zip(merged_batches, spec.vendors)), dim=0
    )
    # breaking down merged batches back to individual batches
    output_data = output_data.reshape(-1, *input_data.shape[2:])

    prev_paths = paths.repeat(1, spec.upstream_sampling).flatten(0).unsqueeze(1)
    new_paths = input_indexes.flatten().unsqueeze(1)
    merged_paths = prev_paths.cat(new_paths, dim=1)

    return output_data, merged_paths
