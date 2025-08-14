import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["text.usetex"] = True


# Define unit cost functions
def unit_cost_all_at_once(N, C):
    """Calculate unit cost for all-at-once approach: U_A = sum(C(i))"""
    return sum(C(i) for i in range(N))


def unit_cost_marketplace(N, M, C):
    """Calculate unit cost for Marketplace approach: U_M = sum(C(i) / M^(N-i-1))"""
    return sum(C(i) / (M ** (N - i - 1)) for i in range(N))


# Define parameters
N_values = np.arange(1, 11)  # Number of layers from 1 to 10
M_values = np.arange(2, 11)  # Number of vendors from 2 to 10
C = lambda i: 1  # Constant computation cost per layer

# Create meshgrid for N and M
N_grid, M_grid = np.meshgrid(N_values, M_values)

# Calculate unit costs
U_M = np.zeros_like(N_grid, dtype=float)
U_A = np.zeros_like(N_grid, dtype=float)
for i in range(N_grid.shape[0]):
    for j in range(N_grid.shape[1]):
        N = int(N_grid[i, j])
        M = int(M_grid[i, j])
        U_M[i, j] = unit_cost_marketplace(N, M, C)
        U_A[i, j] = unit_cost_all_at_once(N, C)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot Marketplace unit cost surface
surf1 = ax.plot_surface(
    N_grid, M_grid, U_M, cmap="viridis", alpha=0.7, label="Marketplace ($$U_M$$)"
)
# Plot All-at-Once unit cost surface
surf2 = ax.plot_surface(
    N_grid, M_grid, U_A, cmap="magma", alpha=0.7, label="All-at-Once ($$U_A$$)"
)

# Add labels and title
ax.set_xlabel("Number of Layers ($N$)")
ax.set_ylabel("Number of Vendors ($M$)")
ax.set_zlabel("Unit Cost")
ax.set_title("Unit Costs: Marketplace vs. All-at-Once")

# Add a color bar for each surface
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5, label="$U_M$")
fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=5, label="$U_A$")

# Set z-axis to log scale for better visualization
ax.set_zscale("log")

# Save the plot
plt.savefig("3d_unit_cost_comparison.png")
plt.close()
