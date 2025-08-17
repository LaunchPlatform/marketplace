# Marketplace
Marketplace is a machine learning experiment project attempting to train a model without using backpropagation on GPU efficiently.
The idea is to breaking down layers of a machine learning model into smaller groups of layers, run them with variants of parameters with different combinations.
We pick the best overall parameters combination and mutate it with different variants of parameters.
To learn more about the idea, please refer to the article:

[Marketplace: my first attempt at training without backprop on GPU efficiently](https://fangpenlin.com/posts/2025/08/18/marketplace-my-first-attempt-at-training-without-backprop-on-gpu-efficiently/)

For example, the beautiful_mnist model included in Tinygrad's example folder can be broken down into three groups of layers:

```python
from marketplace.multi_model import MultiModel, MultiConv2d, MultiLinear, MultiInstanceNorm

[
  Spec(
    model=MultiModel(
      [
        MultiConv2d(vendor_count, 1, 32, 5),
        Tensor.relu,
        MultiConv2d(vendor_count, 32, 32, 5),
        Tensor.relu,
        MultiInstanceNorm(vendor_count, 32),
        Tensor.max_pool2d,
      ]
    )
  ),
  Spec(
    model=MultiModel(
      [
        MultiConv2d(vendor_count, 32, 64, 3),
        Tensor.relu,
        MultiConv2d(vendor_count, 64, 64, 3),
        Tensor.relu,
        MultiInstanceNorm(vendor_count, 64),
        Tensor.max_pool2d,
        lambda x: x.flatten(1),
      ]
    ),
  ),
  Spec(
    model=MultiModel([MultiLinear(vendor_count, 576, 10)]),
  ),
]
```

With that, we can run the model on GPU with different combinations of parameters.
The following code is a simple example of how to run a forward pass of the model.

```python
from tinygrad import Tensor
from tinygrad import TinyJit
from marketplace.multi_model import MultiModelBase
from marketplace.multi_model import Spec
from marketplace.multi_model import forward

@TinyJit
@MultiModelBase.train()
def forward_step() -> tuple[Tensor, Tensor, Tensor]:
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    x = X_train[samples]
    y = Y_train[samples]
    batch_logits, batch_paths = forward(marketplace, x)
    loss = Tensor.stack(
        *(logits.sparse_categorical_crossentropy(y) for logits in batch_logits),
        dim=0,
    )
    best_loss, best_index = loss.topk(1, largest=False)
    best_index = best_index.squeeze(0)
    accuracy = (
        (batch_logits[best_index].sigmoid().argmax(axis=1) == y).sum() / batch_size
    ) * 100
    return (
        best_loss.realize(),
        accuracy.realize(),
        batch_paths[best_index].realize(),
    )

best_loss, best_accuracy, best_path = forward_step()

```

Next, now we know the best parameters combination, we can mutate it with different variants of parameters.

```python
@TinyJit
def mutate_step(best_path: Tensor):
    mutate(
        marketplace=marketplace,
        leading_path=best_path,
        jitter=lr,
    )

mutate_step(best_path)

```

That's it.
We just trained a model without using backpropagation and relying on only the forward pass!
By reepeating the process, we can train a model.
Of course, this is still no match for the backprop training, but it's an interesting start.

## Experiments

All of the experiments are in the `experiments` folder.
To run the training, you can use the following command:

```bash
uv run python -m experiments.beautiful_mnist
```

It comes with some arguments to control the training, you can see them by running:

```bash
uv run python -m experiments.beautiful_mnist --help
```
