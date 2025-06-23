import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ==== Data ====
datasets = ["hb", "icbin", "itodd", "lmo", "tless", "tudl", "ycbv"]
components = ["SAM", "DINO", "Matching"]

# Time values: rows = datasets, cols = components
our_times = np.array(
    [
        [0.614, 0.085, 0.013],
        [0.683, 0.098, 0.014],
        [0.819, 0.058, 0.023],
        [0.753, 0.102, 0.018],
        [0.489, 0.072, 0.012],
        [0.462, 0.077, 0.009],
        [0.932, 0.096, 0.014],
    ]
)

sam6d_times = np.array(
    [
        [2.445, 0.112, 0.020],
        [2.286, 0.099, 0.019],
        [2.413, 0.070, 0.027],
        [2.083, 0.106, 0.030],
        [2.147, 0.114, 0.026],
        [1.845, 0.100, 0.017],
        [2.150, 0.106, 0.018],
    ]
)

# Prepare DataFrame for plotting
df_stacked = pd.DataFrame(
    {
        "Dataset": datasets,
        "Our - SAM": our_times[:, 0],
        "Our - DINO": our_times[:, 1],
        "Our - Matching": our_times[:, 2],
        "SAM6D - SAM": sam6d_times[:, 0],
        "SAM6D - DINO": sam6d_times[:, 1],
        "SAM6D - Matching": sam6d_times[:, 2],
    }
)

# ==== Plot ====
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(datasets))
bar_width = 0.35
group_gap = 0.04

# Component colors
colors = {
    "SAM": "tab:blue",
    "DINO": "tab:orange",
    "Matching": "tab:green",
}

# Plot bars
our_bottom = np.zeros(len(datasets))
sam6d_bottom = np.zeros(len(datasets))

for comp in components:
    our_vals = df_stacked[f"Our - {comp}"]
    sam6d_vals = df_stacked[f"SAM6D - {comp}"]

    x_our = x - (bar_width / 2 + group_gap / 2)

    # For SAM6D (right bar)
    x_sam6d = x + (bar_width / 2 + group_gap / 2)

    # Our method bars (solid)
    ax.bar(x_our, our_vals, bar_width, bottom=our_bottom, color=colors[comp])

    # SAM6D bars (hatched with light gray outline)
    ax.bar(
        x_sam6d,
        sam6d_vals,
        bar_width,
        bottom=sam6d_bottom,
        color=colors[comp],
        hatch="//",
        edgecolor="lightgray",
        linewidth=0,
    )

    for i in range(len(datasets)):
        if comp != "Matching":
            # Inside text for SAM and DINO
            ax.text(
                x_our[i],
                our_bottom[i] + our_vals[i] / 2,
                f"{our_vals[i]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
            ax.text(
                x_sam6d[i],
                sam6d_bottom[i] + sam6d_vals[i] / 2,
                f"{sam6d_vals[i]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
        else:
            # On top text for Matching
            ax.text(
                x_our[i],
                our_bottom[i] + our_vals[i] + 0.015,
                f"{our_vals[i]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )
            ax.text(
                x_sam6d[i],
                sam6d_bottom[i] + sam6d_vals[i] + 0.015,
                f"{sam6d_vals[i]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

    our_bottom += our_vals
    sam6d_bottom += sam6d_vals

# ==== Styling ====
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("Total Runtime (s)")
ax.set_title(
    "Total Runtime per Dataset by Component Our Method (solid) vs. SAM6D (hatched)",
    # pad=14,
    fontsize=16,
)

# Component legend only
legend_elements = [
    Patch(facecolor=colors["SAM"], label="SAM"),
    Patch(facecolor=colors["DINO"], label="DINO"),
    Patch(facecolor=colors["Matching"], label="Matching"),
]
ax.legend(handles=legend_elements, ncol=3, fontsize=9, loc="upper right")

# Aesthetic tweaks
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig("time_components.pdf")
