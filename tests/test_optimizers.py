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
    lr = Tensor(2.0).contiguous().realize()
    optimizer = StochasticOptimizer(
        marketplace=[
            Spec(model=Multiply(3.0), vendor_count=4),
            Spec(model=Multiply(5.0), vendor_count=2),
        ],
        learning_rate=lr,
        seeds=[Tensor([0, 1, 2, 3], dtype=dtypes.uint64).contiguous().realize()],
    )
    assert len(optimizer.delta) == 2
    assert len(optimizer.vendors) == 2

    number_delta0 = optimizer.delta[0]["number"]
    assert number_delta0.shape == (4,)
    # the delta for seed 0 should be all zeros
    assert number_delta0[0].sum().item() == 0
    assert number_delta0[0].min().item() == 0
    assert number_delta0[0].max().item() == 0

    for i in range(1, 4):
        assert number_delta0[i].min().item() > -2.0
        assert number_delta0[i].max().item() < 2.0
