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


# class Model:
#     def __init__(self):
#         self.layers: List[Callable[[Tensor], Tensor]] = [
#             nn.Conv2d(1, 32, 5),
#             Tensor.relu,
#             # nn.Conv2d(32, 32, 5),
#             # Tensor.relu,
#             # nn.BatchNorm(32),
#             # Tensor.max_pool2d,
#             # nn.Conv2d(32, 64, 3),
#             # Tensor.relu,
#             # nn.Conv2d(64, 64, 3),
#             # Tensor.relu,
#             # nn.BatchNorm(64),
#             # Tensor.max_pool2d,
#             # lambda x: x.flatten(1),
#             # nn.Linear(576, 10),
#         ]
#
#     def __call__(self, x: Tensor) -> Tensor:
#         return x.sequential(self.layers)


class Model:
    def __init__(self):
        self.first_layer = [nn.Conv2d(1, 32, 5) for _ in range(100)]

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor.stack(*(m(x) for m in self.first_layer), dim=0)


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
        loss = (
            model(X_train[samples])
            # .sparse_categorical_crossentropy(Y_train[samples])
            # .backward()
        )
        print(loss.size())
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
