import matplotlib.pyplot as plt

# Data
categories = ["LM-O", "T-LESS", "TUD-L", "IC-BIN", "ITODD", "HB", "YCB-V", "YCB-V"]
values = [4.0900755, 5.797638016, 10.77779218, 3.80784144, 2.552189482, 3.8726881, 1.891268335, 3.115150895]

# Font sizes
title_fontsize = 20
label_fontsize = 14
tick_fontsize = 16

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(categories[:-1], values[:-1], color='blue', label='$\\tau_+$=0.6')

# Adjust positions for the last bar to stick near the previous one
bar_width = 0.4
x_positions = [len(categories) - 2 + bar_width]  # Position for the last bar close to the previous
plt.bar(x_positions, [values[-1]], width=bar_width, color='orange', label='$\\tau_+$=0.8')

# Add titles and labels
plt.title('Prompt Quality Ratio of Ours (SG) to the Full-Scene (SAM6D)', fontsize=title_fontsize)
plt.xlabel('Dataset', fontsize=label_fontsize)
# plt.ylabel('Self Grounding / Grid', fontsize=label_fontsize)

# Rotate x-axis labels for better visibility
plt.xticks(list(range(len(categories) - 1)) + x_positions, categories, rotation=45, fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=label_fontsize)

for i, pos in enumerate(x_positions):
    height = [values[-1]][i]
    plt.text(pos, height, f'{height:.2f}', ha='center', va='bottom', fontsize=label_fontsize)

# Add legend
plt.legend(fontsize=label_fontsize)

# Show the plot
plt.tight_layout()
plt.savefig("prompt_quality.pdf")