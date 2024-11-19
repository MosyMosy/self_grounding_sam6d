import matplotlib.pyplot as plt
import numpy as np

# Importing necessary libraries and preparing the updated plot with transparent dotted gray horizontal lines

# Data and font sizes
methods = ['SAM6D', 'SAM6D+CW', 'MPG', 'MPG+CW', 'GS', 'GS+CW']
values = [48.0, 47.8, 48.5, 49.3, 48.4, 49.2]
text_font_size = 18
value_font_size = 16

# Plotting with horizontal dotted lines for better emphasis on levels
plt.figure(figsize=(8, 3))
bar_width = 0.20
x = np.arange(3)

# Plot bars with color, cropping lower limit to emphasize differences
plt.bar(x - bar_width/2, values[::2], width=bar_width, label='Without CW', color='#af4b91')
plt.bar(x + bar_width/2, values[1::2], width=bar_width, label='With CW', color='#41afaa')

# Adding transparent, dotted gray horizontal lines across plot
for y in np.arange(47, 50.1, 0.5):  # Spacing lines every 0.5 units
    plt.axhline(y=y, color='gray', linestyle=':', linewidth=0.75, alpha=0.5)

# Labels and titles with specified font sizes
plt.xlabel('', fontsize=text_font_size)
plt.ylabel('Average Precision', fontsize=text_font_size)
plt.title('Effect of Confidence Weighting (CW) on Methods', fontsize=text_font_size)
plt.xticks(x, ['SAM6D', 'MPG', 'GS'], fontsize=text_font_size)
plt.ylim(47, 50)  # Cropped lower values for emphasis

# Displaying values on top of each bar
for i, v in enumerate(values[::2]):
    plt.text(x[i] - bar_width/2, v + 0.05, f"{v:.1f}", ha='center', fontsize=value_font_size)
for i, v in enumerate(values[1::2]):
    plt.text(x[i] + bar_width/2, v + 0.05, f"{v:.1f}", ha='center', fontsize=value_font_size)

# Legend with font size
plt.legend(fontsize=text_font_size)

# Final adjustments and display
plt.tight_layout()
plt.show()


plt.savefig('cw_effect.pdf')