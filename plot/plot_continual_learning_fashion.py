import json
import logging
import multiprocessing
import pathlib
import signal

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from tinygrad.nn.datasets import mnist

# Simulated image data: list of (image, is_correct) pairs, None for empty cells
logger = logging.getLogger(__name__)

# Save the original default for potential restoration
original_font_size = plt.rcParams["font.size"]
# Scale up font size (e.g., 1.5x the default, which is usually 10pt)
scale_factor = 3.8
plt.rcParams["font.size"] = original_font_size * scale_factor

X_train, _, _, _ = mnist()
X_train = X_train.numpy()

target_new_classes = (3,)
new_X_train, new_Y_train, _, _ = mnist(fashion=True)
class_mask = np.isin(new_Y_train.numpy(), target_new_classes)
target_new_X_train = new_X_train.numpy()[class_mask]
target_new_Y_train = new_Y_train.numpy()[class_mask]


def plot_frame(
    old_images: np.typing.NDArray,
    old_correct: np.typing.NDArray,
    old_learning_accuracy: np.typing.NDArray,
    old_validation_accuracy: np.typing.NDArray,
    old_loss: np.typing.NDArray,
    new_images: np.typing.NDArray,
    new_correct: np.typing.NDArray,
    new_learning_accuracy: np.typing.NDArray,
    new_validation_accuracy: np.typing.NDArray,
    new_loss: np.typing.NDArray,
    steps: np.typing.NDArray,
    output_file: pathlib.Path,
    dpi: int = 50,
):
    images_top = list(
        zip(
            old_images,
            old_correct,
        )
    )
    images_bottom = list(
        zip(
            new_images,
            new_correct,
        )
    )

    # Set up figure with gridspec for images and charts
    fig = plt.figure(figsize=(32, 32))
    fig.suptitle(
        f"Marketplace V2 Continual Learning - Step {steps[-1] + 1}",
        fontsize=48,
    )
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.2, wspace=0.2)

    # Image grid axes
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.set_title(
        f"Old Data ({old_correct.sum()}/{len(old_correct)}, acc={old_learning_accuracy[-1]:.2f}%)"
    )
    ax_bottom = fig.add_subplot(gs[1, 0])
    ax_bottom.set_title(
        f"New Data ({new_correct.sum()}/{len(new_correct)}, acc={new_learning_accuracy[-1]:.2f}%)"
    )

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
                        spine.set_linewidth(4)
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
        steps, old_validation_accuracy, label="Validation Accuracy", color="green"
    )
    ax_loss_top.plot(steps, old_loss, label="Loss", color="red", linestyle="--")
    ax_acc_top.set_title(
        f"Old Data (vacc={old_validation_accuracy[-1]:.2f}, loss={old_loss[-1]:.2f})"
    )
    ax_acc_top.set_xlabel("Steps")
    ax_acc_top.set_ylabel("Accuracy", color="blue")
    ax_acc_top.set_ylim(0, 100)
    ax_loss_top.set_ylabel("Loss", color="red")
    ax_loss_top.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax_acc_top.tick_params(axis="y", colors="blue")
    ax_loss_top.tick_params(axis="y", colors="red")
    ax_acc_top.legend(loc="upper left")
    ax_loss_top.legend(loc="upper right")

    # Plot accuracy and loss for bottom grid
    ax_acc_bottom.plot(
        steps, new_learning_accuracy, label="Learning Accuracy", color="blue"
    )
    ax_acc_bottom.plot(
        steps, new_validation_accuracy, label="Validation Accuracy", color="green"
    )
    ax_loss_bottom.plot(steps, new_loss, label="Loss", color="red", linestyle="--")
    ax_acc_bottom.set_title(
        f"New Data (vacc={new_validation_accuracy[-1]:.2f}, loss={new_loss[-1]:.2f})"
    )
    ax_acc_bottom.set_xlabel("Steps")
    ax_acc_bottom.set_ylabel("Accuracy", color="blue")
    ax_acc_bottom.set_ylim(0, 100)
    ax_loss_bottom.set_ylabel("Loss", color="red")
    ax_loss_bottom.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax_acc_bottom.tick_params(axis="y", colors="blue")
    ax_loss_bottom.tick_params(axis="y", colors="red")
    ax_acc_bottom.legend(loc="upper left")
    ax_loss_bottom.legend(loc="upper right")

    # Remove ticks from image grid axes
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])

    plt.subplots_adjust(left=0.025, right=0.95, bottom=0.05, top=0.925)
    # plt.tight_layout()

    plt.savefig(output_file, dpi=dpi, bbox_inches=None)


@click.command()
@click.argument(
    "INPUT_FILE", type=click.Path(dir_okay=False, exists=True, readable=True)
)
@click.argument(
    "OUTPUT_FOLDER",
    type=click.Path(dir_okay=True, file_okay=False, exists=True, writable=True),
)
@click.option("--limit", type=int)
def main(input_file: str, output_folder: str, limit: int | None):
    steps = []

    old_samples = []
    old_correct = []
    old_learning_accuracy = []
    old_validation_accuracy = []
    old_loss = []

    new_samples = []
    new_correct = []
    new_learning_accuracy = []
    new_validation_accuracy = []
    new_loss = []

    with open(input_file) as replay_file:
        for line in replay_file.readlines():
            data = json.loads(line)

            steps.append(data["global_step"])

            old_learning_accuracy.append(np.array(data["old_correct"]).mean() * 100)
            old_validation_accuracy.append(data["old_test_acc"])
            old_loss.append(data["old_loss"])
            old_correct.append(data["old_correct"])
            old_samples.append(data["old_samples"])

            new_learning_accuracy.append(np.array(data["new_correct"]).mean() * 100)
            new_validation_accuracy.append(data["new_test_acc"])
            new_loss.append(data["new_loss"])
            new_correct.append(data["new_correct"])
            new_samples.append(data["new_samples"])

    old_learning_accuracy = np.array(old_learning_accuracy)
    old_validation_accuracy = np.array(old_validation_accuracy)
    old_loss = np.array(old_loss).mean(axis=1)
    old_correct = np.array(old_correct)
    old_samples = np.array(old_samples)

    new_learning_accuracy = np.array(new_learning_accuracy)
    new_validation_accuracy = np.array(new_validation_accuracy)
    new_loss = np.array(new_loss).mean(axis=1)
    new_correct = np.array(new_correct)
    new_samples = np.array(new_samples)

    steps = np.array(steps)

    def prepare_kwargs(item: tuple[int, int]):
        i, step = item
        count = i + 1
        output_file = pathlib.Path(output_folder) / f"{i}.png"
        logger.info("Writing %s (step %s) to %s", i, step, output_file)
        return dict(
            old_images=X_train[old_samples[i]].reshape(-1, 28, 28),
            old_correct=old_correct[i],
            old_learning_accuracy=old_learning_accuracy[:count],
            old_validation_accuracy=old_validation_accuracy[:count],
            old_loss=old_loss[:count],
            new_images=target_new_X_train[new_samples[i]].reshape(-1, 28, 28),
            new_correct=new_correct[i],
            new_learning_accuracy=new_learning_accuracy[:count],
            new_validation_accuracy=new_validation_accuracy[:count],
            new_loss=new_loss[:count],
            steps=steps[:count],
            output_file=output_file,
        )

    def make_frame(kwargs):
        plot_frame(**kwargs)
        return kwargs["output_file"]

    # ref: https://stackoverflow.com/a/6191991
    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    limited_steps = steps
    if limit is not None:
        limited_steps = steps[:limit]
    with multiprocessing.Pool(16, init_worker) as pool:
        for output_file in pool.imap(
            make_frame,
            filter(
                lambda x: not x["output_file"].exists(),
                map(prepare_kwargs, enumerate(limited_steps)),
            ),
        ):
            logger.info("Wrote to %s", output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
