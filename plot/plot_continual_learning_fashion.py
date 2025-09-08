import json

import matplotlib.pyplot as plt
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.datasets import mnist

# Simulated image data: list of (image, is_correct) pairs, None for empty cells

X_train, Y_train, _, _ = mnist()

target_new_classes = (3,)
new_X_train, new_Y_train, _, _ = mnist(fashion=True)
class_mask = np.isin(new_Y_train.numpy(), target_new_classes)
target_new_X_train = Tensor(new_X_train.numpy()[class_mask])
target_new_Y_train = Tensor(new_Y_train.numpy()[class_mask])


def plot_frame(
    old_samples: np.typing.NDArray,
    old_correct: np.typing.NDArray,
    old_learning_accuracy: np.typing.NDArray,
    old_validation_accuracy: np.typing.NDArray,
    old_loss: np.typing.NDArray,
    new_samples: np.typing.NDArray,
    new_correct: np.typing.NDArray,
    new_learning_accuracy: np.typing.NDArray,
    new_validation_accuracy: np.typing.NDArray,
    new_loss: np.typing.NDArray,
    steps: np.typing.NDArray,
):
    images_top = list(
        zip(
            X_train[Tensor(old_samples)].reshape(-1, 28, 28).numpy(),
            old_correct,
        )
    )
    images_bottom = list(
        zip(
            target_new_X_train[Tensor(new_samples)].reshape(-1, 28, 28).numpy(),
            new_correct,
        )
    )

    # Save the original default for potential restoration
    original_font_size = plt.rcParams["font.size"]

    # Scale up font size (e.g., 1.5x the default, which is usually 10pt)
    scale_factor = 2.0
    plt.rcParams["font.size"] = original_font_size * scale_factor

    # Set up figure with gridspec for images and charts
    fig = plt.figure(figsize=(32, 32))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.1, wspace=0.1)

    # Image grid axes
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.set_title("Old Data")
    ax_bottom = fig.add_subplot(gs[1, 0])
    ax_bottom.set_title("New Data")

    # Chart axes
    ax_acc_top = fig.add_subplot(gs[0, 1])
    ax_loss_top = ax_acc_top.twinx()
    ax_acc_bottom = fig.add_subplot(gs[1, 1])
    ax_loss_bottom = ax_acc_bottom.twinx()

    # Function to plot a single grid
    def plot_grid(ax, images, grid_size=16):
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx >= len(images):
                    sub_ax = ax.inset_axes(
                        [j / grid_size, 1 - (i + 1) / grid_size, 0.05, 0.05]
                    )
                    sub_ax.axis("off")
                    continue
                if images[idx] is not None and images[idx][0] is not None:
                    img, is_correct = images[idx]
                    sub_ax = ax.inset_axes(
                        [j / grid_size, 1 - (i + 1) / grid_size, 0.05, 0.05]
                    )
                    sub_ax.imshow(img, cmap="gray")
                    border_color = "green" if is_correct else "red"
                    for spine in sub_ax.spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(2)
                    sub_ax.set_xticks([])
                    sub_ax.set_yticks([])

    # Plot image grids
    plot_grid(ax_top, images_top)
    plot_grid(ax_bottom, images_bottom)

    # Plot accuracy and loss for top grid
    ax_acc_top.plot(
        steps, old_learning_accuracy, label="Learning Accuracy", color="blue"
    )
    ax_acc_top.plot(
        steps, old_validation_accuracy, label="Validation Accuracy", color="orange"
    )
    ax_loss_top.plot(steps, old_loss, label="Loss", color="red", linestyle="--")
    ax_acc_top.set_title("Old Data: Accuracy and Loss")
    ax_acc_top.set_xlabel("Steps")
    ax_acc_top.set_ylabel("Accuracy", color="blue")
    ax_loss_top.set_ylabel("Loss", color="red")
    ax_acc_top.tick_params(axis="y", colors="blue")
    ax_loss_top.tick_params(axis="y", colors="red")
    ax_acc_top.legend(loc="upper left")
    ax_loss_top.legend(loc="upper right")

    # Plot accuracy and loss for bottom grid
    ax_acc_bottom.plot(
        steps, new_learning_accuracy, label="Learning Accuracy", color="blue"
    )
    ax_acc_bottom.plot(
        steps, new_validation_accuracy, label="Validation Accuracy", color="orange"
    )
    ax_loss_bottom.plot(steps, new_loss, label="Loss", color="red", linestyle="--")
    ax_acc_bottom.set_title("New Data: Accuracy and Loss")
    ax_acc_bottom.set_xlabel("Epoch")
    ax_acc_bottom.set_ylabel("Accuracy", color="blue")
    ax_loss_bottom.set_ylabel("Loss", color="red")
    ax_acc_bottom.tick_params(axis="y", colors="blue")
    ax_loss_bottom.tick_params(axis="y", colors="red")
    ax_acc_bottom.legend(loc="upper left")
    ax_loss_bottom.legend(loc="upper right")

    # Remove ticks from image grid axes
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open("fashion.jsonl") as replay_file:
        for line in replay_file.readlines():
            data = json.loads(line)
            plot_frame(
                old_samples=np.array(data["old_samples"]),
                old_correct=np.array(data["old_correct"]),
                old_learning_accuracy=np.array([30, 33, 34]),
                old_validation_accuracy=np.array([40, 50, 60]),
                old_loss=np.array([0.1, 0.2, 0.3]),
                new_samples=np.array(data["new_samples"]),
                new_correct=np.array(data["new_correct"]),
                new_learning_accuracy=np.array([3, 33, 34]),
                new_validation_accuracy=np.array([40, 70, 60]),
                new_loss=np.array([0.1, 0.2, 0.3]),
                steps=np.array([0, 9, 19]),
            )
            # XXX:
            break
