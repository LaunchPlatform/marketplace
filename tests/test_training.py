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
            Spec(
                vendors=[
                    lambda x: x.leaky_relu(neg_slope) for neg_slope in (0, 0.1, 0.2)
                ]
            ),
            Tensor(
                [
                    [-0.3, 1.0, 2.3],
                    [-0.1, -0.2, -0.3],
                    [1.2, 3.4, 4.5],
                ]
            ),
            (
                Tensor(
                    [
                        [
                            [-0.3, 1.0, 2.3],
                            [-0.1, -0.2, -0.3],
                            [1.2, 3.4, 4.5],
                        ],
                        [
                            [-0.3, 1.0, 2.3],
                            [-0.1, -0.2, -0.3],
                            [1.2, 3.4, 4.5],
                        ],
                        [
                            [-0.3, 1.0, 2.3],
                            [-0.1, -0.2, -0.3],
                            [1.2, 3.4, 4.5],
                        ],
                    ]
                ),
                Tensor([0, 1, 2]),
            ),
        )
    ],
)
def test_produce_with_input_data(
    spec: Spec, x: Tensor, expected: tuple[Tensor, Tensor]
):
    assert list(map(realize, produce(spec=spec, x=x))) == list(map(realize, expected))
