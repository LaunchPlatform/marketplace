import pytest
from tinygrad import Tensor

from marketplace.multi_nn import MultiModelBase
from marketplace.training import produce
from marketplace.training import randperm_skip


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
    "size, skip_index, repeat",
    [
        (10, 0, 100),
    ],
)
def test_randperm_skip(size: int, skip_index: int, repeat: int):
    result = Tensor.stack(
        *[randperm_skip(size, Tensor(skip_index)) for _ in range(repeat)], dim=0
    ).realize()
    assert result.size() == (repeat, size - 1)
    for row in result:
        assert frozenset(row.tolist()) == (
            frozenset(range(size)) - frozenset([skip_index])
        )


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


@pytest.mark.parametrize(
    "model, upstream_sampling, x, paths, leading_vendor_index, leading_input_index",
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
            0,
            1,
        ),
        (
            MultiMultiplyModel([1.0, 3.0, 5.0]),
            3,
            Tensor(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
            Tensor([[0], [1], [2]]),
            1,
            2,
        ),
    ],
)
def test_produce_with_leader_index(
    model: MultiModelBase,
    upstream_sampling: int,
    x: Tensor,
    paths: Tensor,
    leading_vendor_index: int,
    leading_input_index: int,
):
    output, out_paths = produce(
        model=model,
        x=x,
        paths=paths,
        upstream_sampling=upstream_sampling,
        leading_vendor_index=Tensor(leading_vendor_index),
        leading_input_index=Tensor(leading_input_index),
    )

    actual_upstream_sampling = upstream_sampling
    if actual_upstream_sampling == 0:
        actual_upstream_sampling = len(x)

    assert out_paths[leading_vendor_index * actual_upstream_sampling].tolist() == (
        paths[leading_input_index].tolist() + [leading_vendor_index]
    )
    assert (
        output[leading_vendor_index * actual_upstream_sampling].tolist()
        == model(Tensor(leading_vendor_index), x[leading_input_index]).tolist()
    )
    assert all(v >= 0 and v < len(x) for v in out_paths[:, :1].flatten().tolist())
    expected_output = [model(j, x[i]).tolist() for i, j in out_paths[1:]]
    assert output[1:].tolist() == expected_output
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
