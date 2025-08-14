import matplotlib.pyplot as plt
import numpy as np


# Define unit cost functions
def unit_cost_all_at_once(N, C):
    """Calculate unit cost for all-at-once approach: U_A = sum(C(i))"""
    return sum(C(i) for i in range(N))


def unit_cost_marketplace(N, M, C):
    """Calculate unit cost for Marketplace approach: U_M = sum(C(i) / M^(N-i-1))"""
    return sum(C(i) / (M ** (N - i - 1)) for i in range(N))


# Define parameters
M_values = [4, 8, 16, 32, 64]  # Number of vendors per layer
N_values = np.arange(1, 11)  # Number of layers from 1 to 10
C = lambda i: 1  # Constant computation cost per layer

# Initialize lists for plotting
unit_costs_a = []
unit_costs_m = {M: [] for M in M_values}
ratios = {M: [] for M in M_values}

# Calculate unit costs for each N and M
for N in N_values:
    # All-at-once unit cost
    ua = unit_cost_all_at_once(N, C)
    unit_costs_a.append(ua)

    # Marketplace unit cost for each M
    for M in M_values:
        um = unit_cost_marketplace(N, M, C)
        unit_costs_m[M].append(um)
        ratios[M].append(ua / um if um != 0 else float("inf"))

# Create plots
plt.style.use("seaborn-v0_8")

# Plot 1: Unit Cost vs. Number of Layers
plt.figure(figsize=(10, 6))
plt.plot(N_values, unit_costs_a, label="All-at-Once (U_A)", marker="o", linewidth=2)
for M in M_values:
    plt.plot(
        N_values, unit_costs_m[M], label=f"Marketplace (M={M})", marker="s", linewidth=2
    )
plt.xlabel("Number of Layers (N)")
plt.ylabel("Unit Cost")
plt.title("Unit Cost Comparison: All-at-Once vs. Marketplace")
plt.legend()
plt.grid(True)
plt.yscale("log")  # Log scale to better visualize differences
plt.savefig("unit_cost_vs_N.png")
plt.close()

# Plot 2: Unit Cost vs. Number of Vendors for fixed N
fixed_N = [3, 5, 7]  # Different N values to compare
M_range = np.arange(2, 11)
plt.figure(figsize=(10, 6))
for N in fixed_N:
    ua = unit_cost_all_at_once(N, C)
    um_values = [unit_cost_marketplace(N, M, C) for M in M_range]
    plt.plot(M_range, um_values, label=f"Marketplace (N={N})", marker="s", linewidth=2)
    plt.axhline(y=ua, linestyle="--", label=f"All-at-Once (N={N})", alpha=0.7)
plt.xlabel("Number of Vendors per Layer (M)")
plt.ylabel("Unit Cost")
plt.title("Unit Cost vs. Number of Vendors for Fixed N")
plt.legend()
plt.grid(True)
plt.yscale("log")  # Log scale for clarity
plt.savefig("unit_cost_vs_M.png")
plt.close()

# Plot 3: Ratio of Unit Costs vs. Number of Layers
plt.figure(figsize=(10, 6))
for M in M_values:
    plt.plot(N_values, ratios[M], label=f"M={M}", marker="^", linewidth=2)
plt.xlabel("Number of Layers (N)")
plt.ylabel("Efficiency Ratio (U_A / U_M)")
plt.title("Efficiency Ratio: All-at-Once vs. Marketplace")
plt.legend()
plt.grid(True)
plt.yscale("log")  # Log scale to show exponential growth
plt.savefig("efficiency_ratio_vs_N.png")
plt.close()
