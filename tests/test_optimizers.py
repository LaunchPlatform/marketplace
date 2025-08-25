import pytest
from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict

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
    assert len(optimizer.spec_context) == len(optimizer.marketplace)
    # assert len(optimizer.vendors) == len(optimizer.marketplace)


def test_optimizer_schedule_delta_update(optimizer: Optimizer):
    init_seeds = [ctx.seeds.tolist() for ctx in optimizer.spec_context]
    initial_deltas = [
        {key: params.tolist() for key, params in ctx.delta.items()}
        for ctx in optimizer.spec_context
    ]
    for _ in range(10):
        Tensor.realize(*optimizer.schedule_delta_update())
        new_delta = [
            {key: params.tolist() for key, params in ctx.delta.items()}
            for ctx in optimizer.spec_context
        ]
        assert initial_deltas == new_delta
    for _ in range(5):
        for ctx in optimizer.spec_context:
            ctx.seeds.assign(
                Tensor.randint(
                    *ctx.seeds.shape, low=0, high=SEED_MAX, dtype=dtypes.uint64
                )
            ).realize()
        assert [ctx.seeds.tolist() for ctx in optimizer.spec_context] != init_seeds
        last_delta = None
        for _ in range(10):
            Tensor.realize(*optimizer.schedule_delta_update())
            new_delta = [
                {key: params.tolist() for key, params in ctx.delta.items()}
                for ctx in optimizer.spec_context
            ]
            assert new_delta != initial_deltas
            if last_delta is not None:
                assert new_delta == last_delta
            last_delta = new_delta


def test_optimizer_schedule_weight_update(optimizer: Optimizer):
    initial_deltas = [
        {key: params.numpy() for key, params in ctx.delta.items()}
        for ctx in optimizer.spec_context
    ]
    initial_weights = [
        {key: params.numpy() for key, params in get_state_dict(spec.model).items()}
        for spec in optimizer.marketplace
    ]

    # Update with zero seeds, nothing should change
    Tensor.realize(
        *optimizer.schedule_weight_update(
            Tensor.zeros(len(optimizer.marketplace), dtype=dtypes.uint64)
        )
    )
    assert initial_weights == [
        {key: params.numpy() for key, params in get_state_dict(spec.model).items()}
        for spec in optimizer.marketplace
    ]

    # Now the weight should change, but the second should remain the ame
    Tensor.realize(
        *optimizer.schedule_weight_update(Tensor([123, 0, 456], dtype=dtypes.uint64))
    )
    new_weights = [
        {key: params.numpy() for key, params in get_state_dict(spec.model).items()}
        for spec in optimizer.marketplace
    ]
    assert initial_weights[0] != new_weights[0]
    assert initial_weights[1] == new_weights[1]
    assert initial_weights[2] != new_weights[2]
