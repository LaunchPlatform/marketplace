import typing

import pytest
from tinygrad import Tensor

from marketplace.training import produce
from marketplace.training import Spec


class Multiply:
    def __init__(self, value: float):
        self.weight = Tensor(value).contiguous().realize()

    def __call__(self, x: Tensor) -> Tensor:
        return x * self.weight


class MultiplySum:
    def __init__(self, value: float):
        self.weight = Tensor(value).contiguous().realize()

    def __call__(self, x: Tensor):
        return x.sum(axis=1) * self.weight


def realize(x: Tensor) -> list:
    return x.tolist()


@pytest.mark.parametrize(
    "spec, vendors, x, expected",
    [
        (
            Spec(
                model=lambda: None,
                vendor_count=3,
            ),
            [Multiply(v) for v in [0.0, 1.0, 2.0]],
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
        ),
    ],
)
def test_produce_with_input_data(
    spec: Spec,
    vendors: list[typing.Callable],
    x: Tensor,
    expected: tuple[Tensor, Tensor],
):
    assert list(map(realize, produce(spec=spec, vendors=vendors, x=x))) == list(
        map(realize, expected)
    )


@pytest.mark.parametrize(
    "spec, vendors, x, paths",
    [
        (
            Spec(
                model=lambda: None,
                vendor_count=3,
                upstream_sampling=2,
            ),
            [Multiply(v) for v in [1.0, 3.0, 5.0]],
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
            Spec(
                model=lambda: None,
                vendor_count=3,
                upstream_sampling=2,
            ),
            [MultiplySum(v) for v in [1.0, 3.0, 5.0]],
            Tensor(
                [
                    [[1.0, 2.0, 3.0]],
                    [[2.0, 3.0, 4.0]],
                    [[4.0, 5.0, 6.0]],
                ]
            ),
            Tensor([[0], [1], [2]]),
        ),
        (
            Spec(
                model=lambda: None,
                vendor_count=3,
                upstream_sampling=0,
            ),
            [MultiplySum(v) for v in [1.0, 3.0, 5.0]],
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
def test_produce(spec: Spec, vendors: list[typing.Callable], x: Tensor, paths: Tensor):
    output, out_paths = produce(spec=spec, vendors=vendors, x=x, paths=paths)
    assert all(v >= 0 and v < len(x) for v in out_paths[:, :1].flatten().tolist())
    assert (
        out_paths[:, 1:].tolist()
        == (
            Tensor.arange(spec.vendor_count)
            .unsqueeze(1)
            .repeat(1, spec.upstream_sampling if spec.upstream_sampling > 0 else len(x))
            .flatten()
            .unsqueeze(1)
        ).tolist()
    )
    expected_output = [vendors[j.item()](x[i]).tolist() for i, j in out_paths]
    assert output.tolist() == expected_output
