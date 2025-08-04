# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import Callable
from typing import List

from tinygrad import GlobalCounters
from tinygrad import nn
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import colored
from tinygrad.helpers import getenv
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist

from marketplace.training import forward
from marketplace.training import Spec


class Model:
    def __init__(self, layers: List[Callable[[Tensor], Tensor]]):
        self.layers: List[Callable[[Tensor], Tensor]] = layers

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

    marketplace = [
        Spec(
            vendors=[
                Model(
                    [
                        nn.Conv2d(1, 32, 5),
                        Tensor.relu,
                    ]
                )
                for _ in range(10)
            ]
        ),
        Spec(
            vendors=[
                Model(
                    [
                        nn.Conv2d(32, 32, 5),
                        Tensor.relu,
                    ]
                )
                for _ in range(10)
            ],
            upstream_sampling=3,
        ),
        Spec(
            vendors=[Model([nn.BatchNorm(32), Tensor.max_pool2d]) for _ in range(10)],
            upstream_sampling=3,
        ),
        Spec(
            vendors=[
                Model(
                    [
                        nn.Conv2d(32, 64, 3),
                        Tensor.relu,
                        Tensor.relu,
                    ]
                )
                for _ in range(10)
            ],
            upstream_sampling=3,
        ),
        Spec(
            vendors=[
                Model(
                    [
                        nn.Conv2d(64, 64, 3),
                        Tensor.relu,
                    ]
                )
                for _ in range(10)
            ],
            upstream_sampling=3,
        ),
        Spec(
            vendors=[
                Model(
                    [
                        nn.BatchNorm(64),
                        Tensor.max_pool2d,
                    ]
                )
                for _ in range(10)
            ],
            upstream_sampling=3,
        ),
        Spec(
            vendors=[
                Model([lambda x: x.flatten(1), nn.Linear(576, 10)]) for _ in range(10)
            ],
            upstream_sampling=3,
        ),
    ]

    profit_matrix = Tensor.zeros(len(marketplace), 10)

    @TinyJit
    def train_step() -> Tensor:
        global profit_matrix
        samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])

        x = X_train[samples]
        y = Y_train[samples]

        output, paths = forward(marketplace, x)

        profit_attributions = Tensor.stack(
            *(
                (
                    Tensor.zeros(len(marketplace), 10).scatter(
                        dim=1,
                        index=path.unsqueeze(1),
                        src=logits.sparse_categorical_crossentropy(y)
                        .neg()
                        .exp()
                        .repeat(10, 1),
                    )
                )
                for logits, path in zip(output, paths)
            ),
            dim=0,
        ).sum(axis=0)

        print(output.realize().shape, y.shape)
        print(paths.tolist())

        profit_matrix = profit_matrix.add(profit_attributions)
        profit_matrix.realize()

        return output

    #
    # @TinyJit
    # def get_test_acc() -> Tensor:
    #     return (model(X_test).argmax(axis=1) == Y_test).mean() * 100
    #
    test_acc = float("nan")
    for i in (t := trange(getenv("STEPS", 70))):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        loss = train_step()
        # if i % 10 == 9:
        #     test_acc = get_test_acc().item()
        t.set_description(f"loss: {0:6.2f} test_accuracy: {test_acc:5.2f}%")

    print("profit matrix", profit_matrix.numpy())
    # # verify eval acc
    # if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    #     if test_acc >= target and test_acc != 100.0:
    #         print(colored(f"{test_acc=} >= {target}", "green"))
    #     else:
    #         raise ValueError(colored(f"{test_acc=} < {target}", "red"))
