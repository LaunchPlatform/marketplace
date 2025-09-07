import matplotlib.pyplot as plt
import numpy as np

# Simulated data: list of (image, is_correct) pairs, None for empty cells
# Images are 28x28 grayscale for demonstration (replace with your images)
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

# Set up figure with two 16x16 grids stacked vertically
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(16, 32), gridspec_kw={"hspace": 0.2}
)


# Function to plot a single grid
def plot_grid(ax, images, grid_size=16):
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if images[idx] is not None and images[idx][0] is not None:
                img, is_correct = images[idx]
                # Create an axis for each image with some padding
                sub_ax = ax.inset_axes(
                    [j / grid_size, 1 - (i + 1) / grid_size, 0.05, 0.05]
                )
                sub_ax.imshow(img, cmap="gray")
                # Set border color based on prediction
                border_color = "green" if is_correct else "red"
                for spine in sub_ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2)
                sub_ax.set_xticks([])
                sub_ax.set_yticks([])
            else:
                # Create an empty axis for blank cells
                sub_ax = ax.inset_axes(
                    [j / grid_size, 1 - (i + 1) / grid_size, 0.05, 0.05]
                )
                sub_ax.axis("off")


# Plot both grids
plot_grid(ax_top, images_top)
plot_grid(ax_bottom, images_bottom)

# Remove ticks and labels from main axes
ax_top.set_xticks([])
ax_top.set_yticks([])
ax_bottom.set_xticks([])
ax_bottom.set_yticks([])

plt.tight_layout()
plt.show()
