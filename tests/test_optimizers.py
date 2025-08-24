from tinygrad import dtypes
from tinygrad import Tensor

from marketplace.optimizers import StochasticVendor


class Multiply:
    def __init__(self, number: float):
        self.number = Tensor(number).realize()

    def __call__(self, x: Tensor) -> Tensor:
        return x * self.number


def test_stochastic_vendor():
    seed = Tensor(42, dtype=dtypes.uint64).realize()
    lr = Tensor(1e-1).realize()
    vendor = StochasticVendor(seed=seed, learning_rate=lr)
    x = Tensor(4)

    model = Multiply(3.0)
    assert model(x).item() == 12.0

    assert vendor(model)(x).item() == 12.321064949035645

    # ensure that we didn't change the weights of original model
    assert model(x).item() == 12.0


def test_stochastic_vendor_delta_update():
    seed = Tensor(42, dtype=dtypes.uint64).realize()
    lr = Tensor(1e-1).realize()
    vendor = StochasticVendor(seed=seed, learning_rate=lr)
    x = Tensor(4)

    model = Multiply(3.0)
    assert model(x).item() == 12.0

    assert vendor(model)(x).item() == 12.321064949035645
    # ensure counter is reset. with the same seed, the value should be the same
    for _ in range(10):
        seed.assign(Tensor(42, dtype=dtypes.uint64)).realize()
        Tensor.realize(*vendor.schedule_delta_update())
        assert vendor(model)(x).item() == 12.321064949035645

    seed.assign(Tensor(43, dtype=dtypes.uint64)).realize()
    # before update, the delta should remain the same
    assert vendor(model)(x).item() == 12.321064949035645
    Tensor.realize(*vendor.schedule_delta_update())
    assert vendor(model)(x).item() == 12.022363662719727

    lr.assign(Tensor(1e-2)).realize()
    assert vendor(model)(x).item() == 12.022363662719727
    # change lr should also reflect on the delta update
    Tensor.realize(*vendor.schedule_delta_update())
    assert vendor(model)(x).item() == 12.002236366271973
