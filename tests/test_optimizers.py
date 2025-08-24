from tinygrad import Tensor

from marketplace.optimizers import DeltaVendor


class Multiply:
    def __init__(self, number: float):
        self.number = Tensor(number).realize()

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
