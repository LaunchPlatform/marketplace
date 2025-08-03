import dataclasses
import typing

from tinygrad import Tensor


@dataclasses.dataclass
class Spec:
    vendors: list[typing.Callable[[Tensor], Tensor]]
    upstream_sampling: int


def produce(
    spec: Spec, x: Tensor, paths: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    if paths is None:
        # this is the first spec for taking in the raw input, let's feed data to all of them
        data = Tensor.stack(*(vendor(x) for vendor in spec.vendors), dim=0)
        paths = Tensor.arange(len(spec.vendors)).unsqueeze(1)
        return data, paths
    if x.size(0) != paths.size(0):
        raise ValueError(
            "Provided input data's first dimension doesn't match with the paths' first dimension"
        )

    upstream_product_count = paths.size(0)
    upstream_indexes = Tensor.stack(
        *(
            Tensor.randperm(upstream_product_count)[: spec.upstream_sampling]
            for _ in range(len(spec.vendors))
        ),
        dim=0,
    )
