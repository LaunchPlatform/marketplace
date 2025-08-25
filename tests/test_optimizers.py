from tinygrad import dtypes
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
        marketplace=[Spec(model=model, vendor_count=4)],
        learning_rate=lr,
        seeds=[Tensor([0, 1, 2, 3], dtype=dtypes.uint64).contiguous().realize()],
    )
    assert len(optimizer.delta) == 1
    assert len(optimizer.vendors) == 1

    number_delta = optimizer.delta[0]["number"]
    assert number_delta.shape == (4,)
    # the delta for seed 0 should be all zeros
    assert number_delta[0].sum().item() == 0
    assert number_delta[0].min().item() == 0
    assert number_delta[0].max().item() == 0

    for i in range(1, 4):
        assert number_delta[i].min().item() > -2.0
        assert number_delta[i].max().item() < 2.0
