import matplotlib.pyplot as plt
import numpy as np

# Set the font sizes
font_size = 18        # General font size for titles, labels, legends, and data labels
tick_label_size = 16  # Font size for axis numbers (tick labels)

plt.rcParams.update({'font.size': font_size})

# Data
model_sizes = [
    'Small\n(84MB)',    # Added number of parameters
    'Base\n(3308MB)',
    'Large\n(1100MB)',
    'Giant2\n(4200MB)'
]

dinov2_times = [73.298, 26.309, 79.523, 276.331]
dinov2_sg_times = [12.248, 86.125, 104.591, 275.684]

dinov2_memory = [32.661, 65.31, 84.65, 128.34]
dinov2_sg_memory = [35.582, 71.43, 122.34, 140.025]

# Set up the bar positions
x = np.arange(len(model_sizes))  # label locations
width = 0.35  # width of the bars

# Create subplots arranged vertically
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Remove top and right borders (spines) and set tick label font size for both axes
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)

# Plot inference time (Top Plot)
rects1 = ax1.bar(x - width/2, dinov2_times, width, label='DINOv2', color='#1f77b4')
rects2 = ax1.bar(x + width/2, dinov2_sg_times, width, label='DINOv2+SG', color='#ff7f0e')

ax1.set_xlabel('Model Size')
ax1.set_ylabel('Inference Time (ms)')
ax1.set_title('Inference Time Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(model_sizes)
ax1.legend(loc='upper left')

# Add horizontal gridlines
ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

# Function to add data labels above bars
def add_labels(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # Vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=font_size)  # Use general font size

# Add data labels for inference time
add_labels(ax1, rects1)
add_labels(ax1, rects2)

# Plot peak memory usage (Bottom Plot)
rects3 = ax2.bar(x - width/2, dinov2_memory, width, label='DINOv2', color='#aec7e8')
rects4 = ax2.bar(x + width/2, dinov2_sg_memory, width, label='DINOv2+SG', color='#ffbb78')

ax2.set_xlabel('Model Size')
ax2.set_ylabel('Peak Memory Usage (MB)')
ax2.set_title('Peak Memory Usage Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(model_sizes)
ax2.legend(loc='upper left')

# Add horizontal gridlines
ax2.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

# Add data labels for peak memory usage
add_labels(ax2, rects3)
add_labels(ax2, rects4)

# Adjust layout for better fit
fig.tight_layout()

# Save the plot to a file (optional)
# plt.savefig('inference_memory_comparison_vertical_custom.png', dpi=300)

# Display the plot
plt.show()
plt.savefig('inference_memory_comparison.pdf')
