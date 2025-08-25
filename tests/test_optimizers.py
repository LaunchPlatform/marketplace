import pytest
from tinygrad import dtypes
from tinygrad import Tensor

from marketplace.nn import Model
from marketplace.optimizers import DeltaVendor
from marketplace.optimizers import Optimizer
from marketplace.optimizers import SEED_MAX
from marketplace.training import Spec


class Multiply:
    def __init__(self, number: float):
        self.number = Tensor(number).contiguous().realize()

    def __call__(self, x: Tensor) -> Tensor:
        return x * self.number


class Add:
    def __init__(self, number: float):
        self.number = Tensor(number).contiguous().realize()

    def __call__(self, x: Tensor) -> Tensor:
        return x + self.number


@pytest.fixture
def optimizer() -> Optimizer:
    return Optimizer(
        marketplace=[
            Spec(
                model=Model(
                    Multiply(3.0),
                    Add(11.0),
                ),
                vendor_count=4,
            ),
            Spec(
                model=Model(
                    Multiply(6.0),
                    Add(3.0),
                    Multiply(23.0),
                ),
                vendor_count=4,
            ),
            Spec(model=Multiply(5.0), vendor_count=2),
        ],
        learning_rate=Tensor(2.0).contiguous(),
        seeds=[
            Tensor([0, 1, 2, 3], dtype=dtypes.uint64).contiguous(),
            Tensor([0, 1, 2, 3], dtype=dtypes.uint64).contiguous(),
            Tensor([0, 1], dtype=dtypes.uint64).contiguous(),
        ],
    )


def test_delta_vendor():
    model = Model(
        Multiply(3.0),
        Add(7.0),
    )
    vendor = DeltaVendor(
        model=model,
        delta={
            "layers.0.number": Tensor(5.0),
            "layers.1.number": Tensor(2.0),
        },
    )
    x = Tensor(4)
    assert model(x).item() == (x.item() * 3) + 7
    assert vendor(x).item() == (x.item() * (3 + 5)) + (7 + 2)
    # ensure that we didn't change the weights of original model
    assert model(x).item() == (x.item() * 3) + 7

    model.layers[0].number.assign(4.0)
    assert model(x).item() == (x.item() * 4) + 7
    assert vendor(x).item() == (x.item() * (4 + 5)) + (7 + 2)


def test_optimizer(optimizer: Optimizer):
    assert len(optimizer.delta) == len(optimizer.marketplace)
    assert len(optimizer.seeds) == len(optimizer.marketplace)
    assert len(optimizer.vendors) == len(optimizer.marketplace)


def test_optimizer_schedule_delta_update(optimizer: Optimizer):
    materialized_deltas = [
        {key: params.tolist() for key, params in deltas.items()}
        for deltas in optimizer.delta
    ]
    for _ in range(10):
        Tensor.realize(*optimizer.schedule_delta_update())
        new_delta = [
            {key: params.tolist() for key, params in deltas.items()}
            for deltas in optimizer.delta
        ]
        assert materialized_deltas == new_delta
    for _ in range(5):
        for seed in optimizer.seeds:
            seed.assign(
                Tensor.randint(*seed.shape, low=0, high=SEED_MAX, dtype=dtypes.uint64)
            ).realize()
        last_delta = None
        for _ in range(10):
            Tensor.realize(*optimizer.schedule_delta_update())
            new_delta = [
                {key: params.tolist() for key, params in deltas.items()}
                for deltas in optimizer.delta
            ]
            assert new_delta != materialized_deltas
            if last_delta is not None:
                assert new_delta == last_delta
            last_delta = new_delta
