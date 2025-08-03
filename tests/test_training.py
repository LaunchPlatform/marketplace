import functools
import operator

import pytest
from tinygrad import Tensor

from marketplace.training import produce
from marketplace.training import Spec


def realize(x: Tensor) -> list:
    return x.tolist()


@pytest.mark.parametrize(
    "spec, x, expected",
    [
        (
            Spec(vendors=[functools.partial(operator.mul, n) for n in (0.0, 1.0, 2.0)]),
            Tensor([1.0, 2.0, 3.0]),
            (
                Tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0],
                        [2.0, 4.0, 6.0],
                    ]
                ),
                Tensor([[0], [1], [2]]),
            ),
        )
    ],
)
def test_produce_with_input_data(
    spec: Spec, x: Tensor, expected: tuple[Tensor, Tensor]
):
    assert list(map(realize, produce(spec=spec, x=x))) == list(map(realize, expected))


@pytest.mark.parametrize(
    "spec, x, paths, expected",
    [
        (
            Spec(
                vendors=[functools.partial(operator.mul, n) for n in (0.0, 1.0, 2.0)],
                upstream_sampling=2,
            ),
            Tensor(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
            Tensor([[0], [1], [2]]),
            (
                Tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0],
                        [2.0, 4.0, 6.0],
                    ]
                ),
                Tensor([[0], [1], [2]]),
            ),
        )
    ],
)
def test_produce(spec: Spec, x: Tensor, paths: Tensor, expected: tuple[Tensor, Tensor]):
    assert list(map(realize, produce(spec=spec, x=x, paths=paths)))
