import matplotlib.pyplot as plt

# Data
sg_threshold = [0.2, 0.4, 0.6, 0.7, 0.8, 1]
AP = [45.8, 45.8, 46.2, 46.2, 46.3, 45.9]
time = [1.756, 1.395, 0.978, 0.88, 0.873, 1.239]

# Adjustable figure size
figure_width = 8  # Width in inches
figure_height = 2  # Height in inches

# Font sizes
title_fontsize = 24
axis_label_fontsize = 16
tick_label_fontsize = 18

# Plotting
fig, ax1 = plt.subplots(figsize=(figure_width, figure_height))

# Plot average precision
ax1.plot(sg_threshold, AP, marker='o', label='Average Precision', color='blue')
ax1.set_xlabel('', fontsize=axis_label_fontsize)
ax1.set_ylabel('AP', color='blue', fontsize=axis_label_fontsize)
ax1.tick_params(axis='y', labelcolor='blue', labelsize=tick_label_fontsize)
ax1.tick_params(axis='x', labelsize=tick_label_fontsize)

# Create a second y-axis for time
ax2 = ax1.twinx()
ax2.plot(sg_threshold, time, marker='s', label='Time', color='green')
ax2.set_ylabel('Time (s)', color='green', fontsize=axis_label_fontsize)
ax2.tick_params(axis='y', labelcolor='green', labelsize=tick_label_fontsize)

# Title and grid
# plt.title('Average Precision and Time for Different Thresholds', fontsize=title_fontsize)
fig.tight_layout()
plt.grid(True)

# Save the plot
plt.savefig('sg_threshold.pdf', format='pdf', bbox_inches='tight')
