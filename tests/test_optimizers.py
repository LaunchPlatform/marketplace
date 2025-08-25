from tinygrad import Tensor

from marketplace.optimizers import DeltaVendor
from marketplace.optimizers import StochasticOptimizer
from marketplace.training import Spec


class Multiply:
    def __init__(self, number: float):
        self.number = Tensor(number).contiguous().realize()

    def __call__(self, x: Tensor) -> Tensor:
        return x * self.number


def test_delta_vendor():
    model = Multiply(3.0)
    vendor = DeltaVendor(model=model, delta=dict(number=Tensor(5.0)))
    x = Tensor(4)
    assert model(x).item() == 12.0
    assert vendor(x).item() == 32.0
    # ensure that we didn't change the weights of original model
    assert model(x).item() == 12.0

    model.number.assign(4.0)
    assert model(x).item() == 16.0
    assert vendor(x).item() == 36.0


def test_stochastic_optimizer():
    model = Multiply(3.0)
    lr = Tensor(2.0).contiguous().realize()
    optimizer = StochasticOptimizer(
        marketplace=[Spec(model=model, vendor_count=4)], learning_rate=lr
    )
    assert len(optimizer.delta) == 1
    assert len(optimizer.vendors) == 1
    assert len(optimizer.vendors[0]) == 4
