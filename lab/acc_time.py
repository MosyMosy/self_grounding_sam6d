import numpy as np
import matplotlib.pyplot as plt

# Data from the table
methods = ["ZeroPose", "CNOS", "CNOS$^f$", "SAM6D", "NIDS-Net", "SAM6D$^f$", "Ours"]
acc = np.array([36.5, 39.7, 40.4, 48.0, 44.8, 44.8, 49.2])
time = np.array([3.820, 0.220, 1.850, 2.323, 0.49, 0.21, 0.745])
# Beautiful Viridis-inspired colors for each method
colors = ['#5A0000', '#af4b91', '#d7642c', '#e6a532', '#00a0e1', '#466eb4', '#41afaa']




acc_norm = (acc - min(acc)) / (max(acc) - min(acc))
time_norm = (time - min(time)) / (max(time) - min(time)) + 0.5
font_size = 18
axis_font_size = 16

# Bubble sizes based on a weighted acc/time ratio
alpha = 20
bubble_size = [(a / (t)) for a, t in zip(acc_norm, time_norm)]
bubble_size = (bubble_size - min(bubble_size)) / (max(bubble_size) - min(bubble_size)) + 0.5
bubble_size = (bubble_size * alpha)**2.2



# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Add gradient background
n = 100  # Resolution of the gradient
gradient = np.zeros((n, n, 4))  # Create an RGBA gradient

# Fill the gradient from light green (top-left) to light red (bottom-right)
for i in range(n):
    for j in range(n):
        gradient[-i, j, 0] = j / (n - 1) * 0.8  # Red component
        gradient[-i, j, 1] = i / (n - 1) * 0.8  # Green component
        gradient[-i, j, 2] = 0  # No blue component
        gradient[-i, j, 3] = 0.2  # Alpha for transparency

# Plot the gradient background
# ax.imshow(gradient, extent=[min(time) - 0.5, max(time) + 0.5, min(acc) - 2, max(acc) + 2], aspect='auto')

# Plot each method with different colors and bubble sizes
for i, method in enumerate(methods):
    ax.scatter(time[i], acc[i], s=bubble_size[i], color=colors[i], alpha=0.7,  linewidth=0.5, zorder=2)

# Coordinates for SAM6D and Ours
sam6d_time = time[methods.index("SAM6D")]
ours_time = time[methods.index("Ours")]
sam6d_acc = acc[methods.index("SAM6D")]
ours_acc = acc[methods.index("Ours")]

# Draw the horizontal two-way arrow at SAM6D's accuracy level
color_text = '#1b0154'
ax.annotate(
    '', 
    xy=(ours_time, sam6d_acc), 
    xytext=(sam6d_time, sam6d_acc),
    arrowprops=dict(arrowstyle="<->", color=color_text, linestyle='solid', linewidth=1.5)
)

# Add "3X faster" label below the horizontal arrow with background
ax.text((sam6d_time + ours_time) / 2, sam6d_acc - 0.6, "3x Faster", ha='center', fontsize=axis_font_size,
        color=color_text, bbox=dict( alpha=0, edgecolor='none', pad=3))

# Draw the vertical two-way arrow for "1.2%" from "Ours" down to SAM6D
ax.annotate(
    '',
    xy=(ours_time, sam6d_acc),
    xytext=(ours_time, ours_acc),
    arrowprops=dict(arrowstyle="<->", color=color_text, linestyle='solid', linewidth=1.5)
)

# Add "1.2%" label to the left side of the vertical arrow with background
ax.text(ours_time - 0.1, (sam6d_acc + ours_acc) / 2, "1.2%", ha='right', fontsize=axis_font_size,
        color=color_text, bbox=dict(alpha=0, edgecolor='none', pad=3))

# Labels for axes
ax.set_xlabel("Time (seconds)", fontsize=font_size)
ax.set_ylabel("Average Precision", fontsize=font_size)

# Add custom grid with enhanced styling
ax.grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
ax.tick_params(axis='both', labelsize=axis_font_size)

# Add benchmark reference lines for ACC and Time
ax.axhline(40, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
ax.axhline(50, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
ax.axvline(1.0, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
ax.axvline(2.0, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)

# Custom legend for each method with corresponding color
for i, method in enumerate(methods):
    ax.scatter([], [], s=100, color=colors[i], label=method, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.legend(loc="upper right", fontsize=axis_font_size, title="Methods", title_fontsize=axis_font_size)

# Display the plot
plt.tight_layout(pad=1.0)
plt.savefig("acc_time.pdf")
plt.show()
