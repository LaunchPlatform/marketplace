import pytest
from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.helpers import ceildiv
from tinygrad.helpers import prod

from marketplace.random import rand


@pytest.mark.parametrize(
    "shape, seed, counter, expected",
    [
        (
            (6,),
            Tensor(123456, dtype=dtypes.uint64),
            0,
            [
                0.5452135801315308,
                0.28107452392578125,
                0.4398590326309204,
                0.6165577173233032,
                0.04700958728790283,
                0.5229370594024658,
            ],
        ),
        (
            (6,),
            Tensor(123456, dtype=dtypes.uint64),
            2,
            [
                0.4398590326309204,
                0.8822280168533325,
                0.35901951789855957,
                0.5229370594024658,
                0.39503049850463867,
                0.4783148765563965,
            ],
        ),
        (
            (3, 2),
            Tensor(123456, dtype=dtypes.uint64),
            0,
            [
                [0.5452135801315308, 0.28107452392578125],
                [0.4398590326309204, 0.6165577173233032],
                [0.04700958728790283, 0.5229370594024658],
            ],
        ),
    ],
)
def test_rand(shape: tuple[int, ...], seed: Tensor, counter: int, expected: list):
    counter_val = Tensor(counter)
    nums = rand(*shape, seed=seed, counter=counter_val)
    assert nums.tolist() == expected
    assert (
        counter_val.item() == ceildiv(prod(shape) * dtypes.float.itemsize, 4) + counter
    )
