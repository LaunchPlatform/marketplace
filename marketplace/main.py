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

from marketplace.training import Spec


class Model:
    def __init__(self, layers: List[Callable[[Tensor], Tensor]]):
        self.layers: List[Callable[[Tensor], Tensor]] = layers

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


Marketplace = [
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
    )
]


def train():
    pass


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

    model = Model()
    print(nn.state.get_parameters(model)[0].tolist())

    # print(nn.state.get_parameters(model)[1].size())
    # print(nn.state.get_parameters(model)[2].size())
    # print(nn.state.get_parameters(model)[3].size())
    # print(nn.state.get_parameters(model)[4].size())
    # opt = nn.optim.Adam(nn.state.get_parameters(model))
    #
    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
        # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
        loss = model(X_train[samples])
        # .sparse_categorical_crossentropy(Y_train[samples])
        # .backward()
        # )
        # print(loss.size(), idxes.tolist())
        print(loss.realize())
        # opt.step()
        return loss

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

    # # verify eval acc
    # if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    #     if test_acc >= target and test_acc != 100.0:
    #         print(colored(f"{test_acc=} >= {target}", "green"))
    #     else:
    #         raise ValueError(colored(f"{test_acc=} < {target}", "red"))
