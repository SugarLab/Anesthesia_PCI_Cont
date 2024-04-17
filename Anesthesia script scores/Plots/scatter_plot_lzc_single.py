import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set the font to Times New Roman, size 20
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20

# Load your dataset from an Excel file
df = pd.read_excel('/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/statistics/Python_plotting/lzc_single.xlsx')

# Define the desired category order
category_order = ["No experience", "No information", "Experience"]

# Create a categorical type with the specified order
df['Category'] = pd.Categorical(df['Category'], categories=category_order, ordered=True)

# Generate a numerical index for x-axis (experience categories) based on the ordered category
df['experience_index'] = df['Category'].cat.codes

# Define a color map for visual distinction
colors = {'No experience': 'red', 'No information': 'blue', 'Experience': 'green'}

# Create a scatter plot
plt.figure(figsize=(8, 6))

# Add a small jitter to the x-axis positions and plot each category
jitter_strength = 0.05  # Adjust this value as needed for visibility
for category, color in colors.items():
    subset = df[df['Category'] == category]
    jittered_x = subset['experience_index'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(subset))
    plt.scatter(jittered_x, subset['LZ_single_channel'], label=category, color=color, alpha=0.6)

# Adding horizontal lines for specified y values within 0.2 to 0.32
for y in np.arange(0.2, 0.33, 0.01):  # Lines at 0.2, 0.21, ..., 0.32
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Adjust x-axis limits to have equal space on both sides
plt.xlim(-0.5, df['experience_index'].max() + 0.5)

# Adjust the ylim to better fit the data range, now up to 0.32
plt.ylim(0.21, 0.33)

# Customize the x-axis to show category names instead of numerical index
plt.xticks(df['experience_index'].unique(), df['Category'].unique())

# Add labels and legend with Times New Roman font
plt.xlabel('Experience category', labelpad=15)
plt.ylabel('Single channel LZc')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)

# Save the figure in high resolution
plt.savefig('/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/Article_pics/Scatter plots/single_channel_LZc.png', dpi=300, bbox_inches='tight')

plt.show()
