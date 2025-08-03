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
        output = Tensor.stack(*(vendor(x) for vendor in spec.vendors), dim=0)
        paths = Tensor.arange(len(spec.vendors)).unsqueeze(1)
        return output, paths
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
