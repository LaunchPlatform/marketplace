import pytest
from tinygrad import dtypes
from tinygrad import Tensor

from marketplace.random import rand


@pytest.mark.parametrize(
    "shape, seed, base_count, expected",
    [
        (
            (3, 2),
            Tensor([123, 456], dtype=dtypes.uint32),
            0,
            [
                [0.5470587015151978, 0.5954171419143677],
                [0.2251112461090088, 0.45440757274627686],
                [0.040180206298828125, 0.039162516593933105],
            ],
        ),
        (
            (3, 2),
            Tensor([123, 456], dtype=dtypes.uint32),
            1,
            [
                [0.5954171419143677, 0.2251112461090088],
                [0.45440757274627686, 0.040180206298828125],
                [0.039162516593933105, 0.37472009658813477],
            ],
        ),
    ],
)
def test_rand(shape: tuple[int, ...], seed: Tensor, base_count: int, expected: list):
    nums = rand(*shape, seed=seed, base_count=base_count)
    assert nums.tolist() == expected
