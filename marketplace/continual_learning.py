from tinygrad import Tensor

from .optimizers import CachedDeltaVendor
from .training import Spec


def forward_with_paths(
    marketplace: list[Spec],
    x: Tensor,
    paths: Tensor,
    deltas: list[Tensor],
) -> Tensor:
    output = []
    # TODO: this is extremely slow for Tinygrad JIT compiler, should find a better way to do it instead
    for xi, path in zip(x, paths):
        data = xi
        for spec, delta, idx in zip(marketplace, deltas, path):
            vendor = CachedDeltaVendor(
                model=spec.model,
                delta={key: params[idx] for key, params in delta.items()},
            )
            data = vendor(data)
        output.append(data)
    return Tensor.stack(*output, dim=0)
