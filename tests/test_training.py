import pytest
from tinygrad import Tensor

from marketplace.multi_nn import MultiModelBase
from marketplace.training import produce


class MultiMultiplyModel(MultiModelBase):
    def __init__(self, values: list[float]):
        super().__init__()
        self.vendor_count = len(values)
        self.weights = Tensor(values)

    def __call__(self, i: Tensor, x: Tensor):
        return x * self.weights[i]


class MultiMultiplySumModel(MultiModelBase):
    def __init__(self, values: list[float]):
        super().__init__()
        self.vendor_count = len(values)
        self.weights = Tensor(values)

    def __call__(self, i: Tensor, x: Tensor):
        return x.sum(axis=1) * self.weights[i]


def realize(x: Tensor) -> list:
    return x.tolist()


@pytest.mark.parametrize(
    "model, x, expected",
    [
        (
            MultiMultiplyModel([0.0, 1.0, 2.0]),
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
    model: MultiModelBase, x: Tensor, expected: tuple[Tensor, Tensor]
):
    assert list(map(realize, produce(model=model, x=x))) == list(map(realize, expected))


@pytest.mark.parametrize(
    "model, upstream_sampling, x, paths",
    [
        (
            MultiMultiplyModel([1.0, 3.0, 5.0]),
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
            MultiMultiplySumModel([1.0, 3.0, 5.0]),
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
        (
            MultiMultiplyModel([1.0, 3.0, 5.0]),
            0,
            Tensor(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
            Tensor([[0], [1], [2]]),
        ),
    ],
)
def test_produce(
    model: MultiModelBase, upstream_sampling: int, x: Tensor, paths: Tensor
):
    output, out_paths = produce(
        model=model, x=x, paths=paths, upstream_sampling=upstream_sampling
    )

    assert all(v >= 0 and v < len(x) for v in out_paths[:, :1].flatten().tolist())
    assert (
        out_paths[:, 1:].tolist()
        == (
            Tensor.arange(model.vendor_count)
            .unsqueeze(1)
            .repeat(1, upstream_sampling if upstream_sampling > 0 else len(x))
            .flatten()
            .unsqueeze(1)
        ).tolist()
    )

    expected_output = [model(j, x[i]).tolist() for i, j in out_paths]
    assert output.tolist() == expected_output
