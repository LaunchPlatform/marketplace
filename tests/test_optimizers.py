from tinygrad import dtypes
from tinygrad import Tensor

from marketplace.optimizers import StochasticVendor


class Multiply:
    def __init__(self, number: float):
        self.number = Tensor(number).realize()

    def __call__(self, x: Tensor) -> Tensor:
        return x * self.number


def test_stochastic_vendor():
    vendor = StochasticVendor(
        seed=Tensor(42, dtype=dtypes.uint64), learning_rate=Tensor(1e-1)
    )

    model = Multiply(3.0)
    assert model(Tensor(4)).item() == 12.0

    vendored = vendor(model)
    assert vendored(Tensor(4)).item() == 12.321065902709961

    # ensure that we didn't change the weights of original model
    assert model(Tensor(4)).item() == 12.0
