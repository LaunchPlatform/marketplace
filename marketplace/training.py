import dataclasses
import typing

import tinygrad


@dataclasses.dataclass
class Spec:
    models: list[typing.Callable[[tinygrad.Tensor], tinygrad.Tensor]]
    vendors: int
    upstream_sampling: int


def forward(
    spec: Spec, data: tinygrad.Tensor, paths: tinygrad.Tensor | None = None
) -> tuple[tinygrad.Tensor, tinygrad.Tensor]:
    pass
