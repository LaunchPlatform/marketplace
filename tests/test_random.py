import pytest
from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.dtype import DTypeLike
from tinygrad.helpers import ceildiv
from tinygrad.helpers import prod

from marketplace.random import counter_advance
from marketplace.random import rand
from marketplace.random import RandomNumberGenerator


@pytest.fixture
def rng() -> RandomNumberGenerator:
    return RandomNumberGenerator(
        seed=Tensor(0, dtype=dtypes.uint64), counter=Tensor(0, dtype=dtypes.uint)
    )


@pytest.mark.parametrize(
    "shape, seed, counter, expected",
    [
        (
            (),
            Tensor(123456, dtype=dtypes.uint64),
            0,
            0.7353423833847046,
        ),
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
def test_rand(
    shape: tuple[int, ...], seed: Tensor, counter: int, expected: list | float
):
    counter_val = Tensor(counter)
    nums = rand(*shape, seed=seed, counter=counter_val)
    if isinstance(expected, list):
        assert nums.tolist() == expected
    else:
        assert nums.item() == expected
    assert (
        counter_val.item() == ceildiv(prod(shape) * dtypes.float.itemsize, 4) + counter
    )


def test_rng_rand(rng: RandomNumberGenerator):
    random_numbers = rng.rand(512, 768).realize()
    assert random_numbers.min().item() >= 0.0
    assert random_numbers.max().item() < 1.0
    assert random_numbers.mean().item() == pytest.approx(0.5, rel=1e-03)


def test_rng_uniform(rng: RandomNumberGenerator):
    random_numbers = rng.uniform(512, 768, low=0, high=10).realize()
    assert random_numbers.min().item() >= 0.0
    assert random_numbers.max().item() < 10.0
    assert random_numbers.mean().item() == pytest.approx(5, rel=1e-03)


@pytest.mark.parametrize(
    "shape, dtype, expected",
    [
        ((0,), dtypes.float, 0),
        ((1, 2, 3), dtypes.float, 6),
        ((1, 2, 3), dtypes.float16, 3),
        ((5,), dtypes.float16, 3),
    ],
)
def test_counter_advance(
    shape: tuple[int, ...], dtype: DTypeLike | None, expected: int
):
    assert counter_advance(*shape, dtype=dtype) == expected
