import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


# Updated data with new accuracy and time values for each similarity threshold
similarity_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
acc_values = [46.8, 46.8, 46.8, 46.9, 47.1, 46.3, 39.4]
times = [2.556, 2.256, 1.87, 1.369, 0.862, 0.605, 0.432]


# Plotting the data with distinct colors for time and ACC, and a marker at the selected threshold (0.6)
# Define font and axis sizes
font_size = 18
axis_size = 14

# Define colors for each chart and selected threshold
time_color = "#466eb4"
acc_color = "#d7642c"
current_color = "#41afaa"

# Plotting the data with specified font sizes and colors
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 4))

# Plot time on the upper chart
ax1.plot(similarity_thresholds, times, marker="o", linestyle="-", color=time_color, label="Time (s)")
ax1.set_ylabel("Time (s)", color=time_color, fontsize=font_size)
ax1.axvline(0.6, color=current_color, linestyle="--")  # Mark the selected threshold at 0.6
ax1.plot(0.6, 0.862, "o", color="gold")  # Marker at the selected point for time
ax1.set_title("Time and AP vs Similarity Threshold", fontsize=font_size)
ax1.tick_params(axis='both', labelsize=axis_size)
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax1.grid(True)

# Plot ACC on the lower chart
ax2.plot(similarity_thresholds, acc_values, marker="o", linestyle="-", color=acc_color, label="Accuracy (AP)")
ax2.set_xlabel("Similarity Threshold (τ+)", fontsize=font_size)
ax2.set_ylabel("AP", color=acc_color, fontsize=font_size)
ax2.axvline(0.6, color=current_color, linestyle="--")  # Mark the selected threshold at 0.6
ax2.plot(0.6, 47.1, "o", color="gold")  # Marker at the selected point for ACC
ax2.tick_params(axis='both', labelsize=axis_size)
ax2.set_ylim(38, 48)
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax2.grid(True)

# Save and display the plot
plt.tight_layout()
plt.savefig("sim_threshold.pdf")
plt.show()

