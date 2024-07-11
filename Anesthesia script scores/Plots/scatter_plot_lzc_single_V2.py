import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

# Set the font to Times New Roman, size 20
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20

# Load your dataset from an Excel file
df = pd.read_excel('/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/statistics/Python_plotting/lzc_single_V2.xlsx')

# Define the desired category order
category_order = ["No experience", "No information", "Experience"]

# Create a categorical type with the specified order
df['Category'] = pd.Categorical(df['Category'], categories=category_order, ordered=True)

# Generate unique colors for each unique Subject ID using a color map
unique_ids = df['Subject_ID'].unique()
# Create a color palette with enough colors
color_palette = sns.color_palette("hsv", len(unique_ids))
colors = {id: color_palette[i] for i, id in enumerate(unique_ids)}

# Add color to DataFrame for consistent access
df['color'] = df['Subject_ID'].apply(lambda x: colors[x])

# Create a swarm plot with proper color mapping
plt.figure(figsize=(8, 6))
sns.swarmplot(x='Category', y='LZ_single_channel', data=df, hue='Subject_ID', palette=colors)

# Adding horizontal lines for specified y values within the range
for y in np.arange(0.2, 0.33, 0.01):
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Adjust the ylim to better fit the data range, now up to 0.32
plt.ylim(0.21, 0.33)

# Customize the x-axis to show category names
plt.xlabel('Experience category', labelpad=15)
plt.ylabel('Single channel LZc')
plt.title('Swarm Plot of LZc by Experience Category')

# Handle legend if needed
plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Save the figure in high resolution
plt.savefig('/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/Article_pics/Scatter plots/single_channel_LZc_V2.png', dpi=300, bbox_inches='tight')

plt.show()
