import matplotlib.pyplot as plt

# Data
datasets = ["LM-O", "T-Less", "TUD-L", "IC-BIN", "ITODD", "HB", "YCB-V"]
template_counts = [1, 2, 4, 8, 16, 32, 42]
values = [
    [0.471, 0.472, 0.473, 0.472, 0.472, 0.472, 0.471],  # LM-O
    [0.462, 0.462, 0.463, 0.462, 0.460, 0.462, 0.462],  # T-Less
    [0.584, 0.586, 0.583, 0.585, 0.586, 0.587, 0.586],  # TUD-L
    [0.278, 0.380, 0.377, 0.380, 0.378, 0.378, 0.379],  # IC-BIN
    [0.342, 0.355, 0.355, 0.349, 0.349, 0.352, 0.352],  # ITODD
    [0.560, 0.570, 0.568, 0.569, 0.573, 0.580, 0.576],  # HB
    [0.620, 0.619, 0.619, 0.619, 0.619, 0.617, 0.618],  # YCB-V
]

# Custom color list for datasets
custom_colors = ["#FF0000", "#00FF00", "#0000FF", "#FF00FF", "#00FFFF", "#800080", "#FFA500"]

# Font sizes for customization
title_fontsize = 20
label_fontsize = 16
legend_fontsize = 14

# Plot with customizations
plt.figure(figsize=(12, 6))
for i, dataset in enumerate(datasets):
    plt.plot(template_counts, values[i], marker='.', label=dataset, color=custom_colors[i], zorder=2)
    # Mark peak value
    peak_idx = values[i].index(max(values[i]))
    plt.scatter(template_counts[peak_idx], values[i][peak_idx], color="#FFD700", marker='*', s=150, zorder=3)

# Add star legend entry only once
plt.scatter([], [], color="#FFD700", marker='*', s=150, label="Peak Value")

plt.title("Average Precision vs Number of Views ($|C|$)", fontsize=title_fontsize)
plt.xlabel("Number of Templates", fontsize=label_fontsize)
plt.ylabel("Average Precision", fontsize=label_fontsize)
plt.xticks(template_counts)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legend_fontsize)
plt.grid(alpha=0.5, zorder=1)
plt.tight_layout()
plt.savefig("num_templates.pdf")
