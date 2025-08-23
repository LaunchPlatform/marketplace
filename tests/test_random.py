import pytest
from tinygrad import dtypes
from tinygrad import Tensor

from marketplace.random import rand


@pytest.mark.parametrize(
    "shape, seed, base_count, expected",
    [
        (
            (5,),
            Tensor(123456, dtype=dtypes.uint64),
            0,
            [
                0.5452135801315308,
                0.28107452392578125,
                0.4398590326309204,
                0.6165577173233032,
                0.04700958728790283,
            ],
        ),
        (
            (5,),
            Tensor(123456, dtype=dtypes.uint64),
            2,
            [
                0.4398590326309204,
                0.8822280168533325,
                0.35901951789855957,
                0.5229370594024658,
                0.39503049850463867,
            ],
        ),
    ],
)
def test_rand(shape: tuple[int, ...], seed: Tensor, base_count: int, expected: list):
    nums = rand(*shape, seed=seed, base_count=base_count)
    print(nums.tolist())
    assert nums.tolist() == expected
