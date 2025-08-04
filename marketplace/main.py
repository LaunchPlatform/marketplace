# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
import time
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
from marketplace.training import uniform_between


class Model:
    def __init__(self, layers: List[Callable[[Tensor], Tensor]]):
        self.layers: List[Callable[[Tensor], Tensor]] = layers

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

    VENDOR_COUNT = 16
    UPSTREAM_SAMPLING = 4
    OFFSPRING_COUNT = 8

    marketplace = [
        Spec(
            vendors=[
                Model(
                    [
                        nn.Conv2d(1, 32, 5),
                        Tensor.relu,
                    ]
                )
                for _ in range(VENDOR_COUNT)
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
                for _ in range(VENDOR_COUNT)
            ],
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
        Spec(
            vendors=[
                Model([nn.BatchNorm(32), Tensor.max_pool2d])
                for _ in range(VENDOR_COUNT)
            ],
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
        Spec(
            vendors=[
                Model(
                    [
                        nn.Conv2d(32, 64, 3),
                        Tensor.relu,
                    ]
                )
                for _ in range(VENDOR_COUNT)
            ],
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
        Spec(
            vendors=[
                Model(
                    [
                        nn.Conv2d(64, 64, 3),
                        Tensor.relu,
                    ]
                )
                for _ in range(VENDOR_COUNT)
            ],
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
        Spec(
            vendors=[
                Model(
                    [
                        nn.BatchNorm(64),
                        Tensor.max_pool2d,
                    ]
                )
                for _ in range(VENDOR_COUNT)
            ],
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
        Spec(
            vendors=[
                Model([lambda x: x.flatten(1), nn.Linear(576, 10)])
                for _ in range(VENDOR_COUNT)
            ],
            upstream_sampling=UPSTREAM_SAMPLING,
        ),
    ]

    @TinyJit
    def train_step() -> Tensor:
        samples = Tensor.randint(getenv("BS", 128), high=X_train.shape[0])

        x = X_train[samples]
        y = Y_train[samples]

        output, paths = forward(marketplace, x)

        profit_matrix = Tensor.stack(
            *(
                (
                    Tensor.zeros(len(marketplace), VENDOR_COUNT).scatter(
                        dim=1,
                        index=path.unsqueeze(1),
                        src=logits.sparse_categorical_crossentropy(y)
                        .neg()
                        .exp()
                        .repeat(VENDOR_COUNT, 1),
                    )
                )
                for logits, path in zip(output, paths)
            ),
            dim=0,
        ).sum(axis=0)

        for vendor_profits, spec in zip(profit_matrix, marketplace):
            reproduce_matrix = (
                vendor_profits.reshape(-1, 1) * vendor_profits.reshape(1, -1)
            ).triu(diagonal=1)
            parent_indexes = reproduce_matrix.flatten().multinomial(
                OFFSPRING_COUNT, replacement=True
            )
            lhs_indexes = parent_indexes // vendor_profits.shape[0]
            rhs_indexes = parent_indexes % vendor_profits.shape[0]
            for lhs_idx, rhs_idx in zip(lhs_indexes, rhs_indexes):
                lhs = spec.vendors[lhs_idx.item()]
                rhs = spec.vendors[rhs_idx.item()]
                lhs_params = nn.state.get_state_dict(lhs)
                rhs_params = nn.state.get_state_dict(rhs)
                new_params = {
                    uniform_between(
                        lhs=lhs_params[key],
                        rhs=rhs_params[key],
                    ).realize()
                    for key in lhs_params
                }
                print(new_params)

        #
        # profit_matrix.realize()
        #
        # reproduce_count = VENDOR_COUNT - PHASE_OUT_COUNT
        # reproduce_weights, reproduce_indexes = profit_matrix.topk(
        #     reproduce_count, dim=1,
        # )
        # print("$" * 10, reproduce_weights.tolist())
        # # print("@" * 10, phase_out_indexes.tolist())
        #
        # print("### reproduce_weights", reproduce_weights[0].multinomial(reproduce_count, replacement=True).tolist())

        return output

    #
    # @TinyJit
    # def get_test_acc() -> Tensor:
    #     return (model(X_test).argmax(axis=1) == Y_test).mean() * 100
    #
    test_acc = float("nan")
    for i in (t := trange(getenv("STEPS", 1000))):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        start_time = time.perf_counter()
        loss = train_step()
        end_time = time.perf_counter()
        run_time = end_time - start_time
        # if i % 10 == 9:
        #     test_acc = get_test_acc().item()
        t.set_description(
            f"loss: {0:6.2f}, {GlobalCounters.global_ops * 1e-9 / run_time:9.2f} GFLOPS"
        )

    # print("profit matrix", profit_matrix.numpy())
    # # verify eval acc
    # if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    #     if test_acc >= target and test_acc != 100.0:
    #         print(colored(f"{test_acc=} >= {target}", "green"))
    #     else:
    #         raise ValueError(colored(f"{test_acc=} < {target}", "red"))
