import functools
import operator

import pytest
from tinygrad import Tensor

from marketplace.training import Model
from marketplace.training import produce
from marketplace.training import uniform_between


def realize(x: Tensor) -> list:
    return x.tolist()


@pytest.mark.parametrize(
    "vendors, x, expected",
    [
        (
            [functools.partial(operator.mul, n) for n in (0.0, 1.0, 2.0)],
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
    vendors: list[Model], x: Tensor, expected: tuple[Tensor, Tensor]
):
    assert list(map(realize, produce(vendors=vendors, x=x))) == list(
        map(realize, expected)
    )


@pytest.mark.parametrize(
    "vendors, upstream_sampling, x, paths",
    [
        (
            [functools.partial(operator.mul, n) for n in (1.0, 3.0, 5.0)],
            2,
            Tensor(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
            Tensor([[0], [1], [2]]),
        ),
        (
            [lambda x, n=n: x.sum(axis=1) * n for n in (1.0, 3.0, 5.0)],
            2,
            Tensor(
                [
                    [[1.0, 2.0, 3.0]],
                    [[2.0, 3.0, 4.0]],
                    [[4.0, 5.0, 6.0]],
                ]
            ),
            Tensor([[0], [1], [2]]),
        ),
    ],
)
def test_produce(
    vendors: list[Model], upstream_sampling: int, x: Tensor, paths: Tensor
):
    output, out_paths = produce(
        vendors=vendors, x=x, paths=paths, upstream_sampling=upstream_sampling
    )

    assert all(v >= 0 and v < len(x) for v in out_paths[:, :1].flatten().tolist())
    assert (
        out_paths[:, 1:].tolist()
        == (
            Tensor.arange(len(vendors))
            .unsqueeze(1)
            .repeat(1, upstream_sampling)
            .flatten()
            .unsqueeze(1)
        ).tolist()
    )

    expected_output = [vendors[j.item()](x[i]).tolist() for i, j in out_paths]
    assert output.tolist() == expected_output


@pytest.mark.parametrize(
    "lhs, rhs, jitter_scale, jitter_offset, expected",
    [
        (
            Tensor.zeros(100, 100, 100),
            Tensor.ones(100, 100, 100),
            None,
            None,
            (0.0, 1.0),
        ),
        (
            Tensor.zeros(100, 100, 100),
            Tensor.zeros(100, 100, 100),
            None,
            None,
            (0.0, 0.0),
        ),
        (
            Tensor.ones(100, 100, 100),
            Tensor.ones(100, 100, 100),
            None,
            None,
            (1.0, 1.0),
        ),
        (
            Tensor.zeros(100, 100, 100),
            Tensor.ones(100, 100, 100),
            Tensor(0.1),
            None,
            (-0.1, 1.1),
        ),
        (
            Tensor.zeros(100, 100, 100),
            Tensor.ones(100, 100, 100),
            Tensor(0.1),
            Tensor(0.025),
            (-0.125, 1.125),
        ),
    ],
)
def test_uniform_between(
    lhs: Tensor,
    rhs: Tensor,
    jitter_scale: Tensor | None,
    jitter_offset: Tensor | None,
    expected: tuple[float, float],
):
    res = uniform_between(
        lhs=lhs, rhs=rhs, jitter_scale=jitter_scale, jitter_offset=jitter_offset
    )
    assert res.min().item() >= expected[0]
    assert res.max().item() <= expected[1]
    assert res.mean().item() == pytest.approx((expected[0] + expected[1]) / 2, 0.01)
    # TODO: add Kolmogorov-Smirnov test test if needed
