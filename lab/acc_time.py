import numpy as np
import matplotlib.pyplot as plt

# Data from the table
methods = ["ZeroPose", "CNOS", "CNOS_f", "SAM6D", "SAM6D_f", "Ours"]
acc = [36.5, 39.7, 40.4, 48.0, 44.8, 49.2]
time = [3.820, 0.220, 1.850, 2.323, 0.21, 0.745]

font_size = 16
bubble_size = [(x - 30)**2 * 1 for x in acc]

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
        gradient[-i, j, 3] = 0.3  # Alpha for transparency

# Plot the gradient background
# ax.imshow(gradient, extent=[min(time) - 0.5, max(time) + 0.5, min(acc) - 2, max(acc) + 2], aspect='auto')

# Plot black circles for other methods and a blue cross for "Ours"
methods_display = ["ZeroPose", "CNOS", "CNOS$^f$", "SAM6D", "SAM6D$^f$", "Ours"]
for i, method in enumerate(methods_display):
    if method == "Ours":
        ax.scatter(time[i], acc[i], s=50, color='black', marker='x', label="Ours")  # Blue cross for "Ours"
        ax.text(time[i] + 0.05, acc[i], method, fontsize=font_size, va='center', color='black')
    elif method == "ZeroPose":
        ax.scatter(time[i], acc[i], s=50, color='black', marker='o')  # Black circles
        ax.text(time[i] - 0.65, acc[i], method, fontsize=font_size, va='center', color='black')
    else:
        ax.scatter(time[i], acc[i], s=50, color='black', marker='o')  # Black circles
        ax.text(time[i] + 0.05, acc[i], method, fontsize=font_size, va='center', color='black')

# Coordinates for SAM6D and Ours
sam6d_time = time[methods.index("SAM6D")]
ours_time = time[methods.index("Ours")]
sam6d_acc = acc[methods.index("SAM6D")]
ours_acc = acc[methods.index("Ours")]

# Draw the horizontal two-way arrow at SAM6D's accuracy level
ax.annotate(
    '', 
    xy=(ours_time, sam6d_acc), 
    xytext=(sam6d_time, sam6d_acc),
    arrowprops=dict(arrowstyle="<->", color='steelblue', linestyle='solid', linewidth=1)
)

# Add "3X faster" label below the horizontal arrow
ax.text((sam6d_time + ours_time) / 2, sam6d_acc - 0.6, "3X faster", ha='center', fontsize=font_size, color='steelblue')

# Draw the vertical two-way arrow for "1.2%" from "Ours" down to SAM6D
ax.annotate(
    '',
    xy=(ours_time, sam6d_acc),
    xytext=(ours_time, ours_acc),
    arrowprops=dict(arrowstyle="<->", color='steelblue', linestyle='solid', linewidth=1)
)

# Add "1.2%" label to the left side of the vertical arrow
ax.text(ours_time - 0.05, (sam6d_acc + ours_acc) / 2, "1.2%", ha='right', fontsize=font_size, color='steelblue')

# Labels for axes
ax.set_xlabel("Time (seconds)", fontsize=font_size)
ax.set_ylabel("Mean ACC", fontsize=font_size)

# Remove the grid
# ax.grid(False)
ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.tick_params(axis='both', labelsize=12)


# Display the plot
plt.tight_layout(pad=1.0)
plt.savefig("acc_time.pdf")
plt.show()
