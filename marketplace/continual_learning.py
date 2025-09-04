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
    for xi, i, delta in zip(x, paths, deltas):
        data = xi
        for spec in marketplace:
            vendor = CachedDeltaVendor(model=spec.model, delta=delta[i])
            data = vendor(data)
        output.append(data)
    return Tensor.stack(*output, dim=0)
