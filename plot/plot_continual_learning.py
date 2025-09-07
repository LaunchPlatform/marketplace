import matplotlib.pyplot as plt
import numpy as np

# Simulated image data: list of (image, is_correct) pairs, None for empty cells
np.random.seed(42)
images_top = [
    (
        np.random.rand(28, 28),
        np.random.choice([True, False]) if np.random.rand() > 0.2 else None,
    )
    for _ in range(256)
]
images_bottom = [
    (
        np.random.rand(28, 28),
        np.random.choice([True, False]) if np.random.rand() > 0.2 else None,
    )
    for _ in range(256)
]

# Simulated accuracy and loss data
epochs = np.arange(1, 21)
accuracy_train_top = 0.5 + 0.4 * (1 - np.exp(-epochs / 10))
accuracy_val_top = 0.48 + 0.38 * (1 - np.exp(-epochs / 8))
loss_top = 2.0 * np.exp(-epochs / 5)
accuracy_train_bottom = 0.45 + 0.35 * (1 - np.exp(-epochs / 12))
accuracy_val_bottom = 0.43 + 0.33 * (1 - np.exp(-epochs / 10))
loss_bottom = 2.5 * np.exp(-epochs / 4)

# Set up figure with gridspec for images and charts
fig = plt.figure(figsize=(22, 32))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0.2, wspace=0.1)

# Image grid axes
ax_top = fig.add_subplot(gs[0, 0])
ax_bottom = fig.add_subplot(gs[1, 0])

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
            else:
                sub_ax = ax.inset_axes(
                    [j / grid_size, 1 - (i + 1) / grid_size, 0.05, 0.05]
                )
                sub_ax.axis("off")


# Plot image grids
plot_grid(ax_top, images_top)
plot_grid(ax_bottom, images_bottom)

# Plot accuracy and loss for top grid
ax_acc_top.plot(epochs, accuracy_train_top, label="Training Accuracy", color="blue")
ax_acc_top.plot(epochs, accuracy_val_top, label="Validation Accuracy", color="orange")
ax_loss_top.plot(epochs, loss_top, label="Loss", color="red", linestyle="--")
ax_acc_top.set_title("Top Grid: Accuracy and Loss")
ax_acc_top.set_xlabel("Epoch")
ax_acc_top.set_ylabel("Accuracy", color="blue")
ax_loss_top.set_ylabel("Loss", color="red")
ax_acc_top.tick_params(axis="y", colors="blue")
ax_loss_top.tick_params(axis="y", colors="red")
ax_acc_top.legend(loc="upper left")
ax_loss_top.legend(loc="upper right")

# Plot accuracy and loss for bottom grid
ax_acc_bottom.plot(
    epochs, accuracy_train_bottom, label="Training Accuracy", color="blue"
)
ax_acc_bottom.plot(
    epochs, accuracy_val_bottom, label="Validation Accuracy", color="orange"
)
ax_loss_bottom.plot(epochs, loss_bottom, label="Loss", color="red", linestyle="--")
ax_acc_bottom.set_title("Bottom Grid: Accuracy and Loss")
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

plt.tight_layout()
plt.show()
