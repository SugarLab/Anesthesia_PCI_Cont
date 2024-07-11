import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set the font to Times New Roman, size 20
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20

# Load your dataset from an Excel file
df = pd.read_excel('/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/statistics/Python_plotting/pcist_V2.xlsx')

# Define the desired category order
category_order = ["No experience", "No information", "Experience"]

# Create a categorical type with the specified order
df['Category'] = pd.Categorical(df['Category'], categories=category_order, ordered=True)

# Generate a numerical index for x-axis (experience categories) based on the ordered category
df['experience_index'] = df['Category'].cat.codes

# Generate unique colors for each unique Subject ID using a color map
unique_ids = df['Subject_ID'].unique()
colors = {id: plt.cm.tab20(i % 20) for i, id in enumerate(unique_ids)}

# Create a scatter plot
plt.figure(figsize=(8, 6))

# Add a small jitter to the x-axis positions and plot each subject ID
jitter_strength = 0.05  # Adjust this value as needed for visibility
for id in unique_ids:
    subset = df[df['Subject_ID'] == id]
    jittered_x = subset['experience_index'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(subset))
    plt.scatter(jittered_x, subset['PCIst'], label=id, color=colors[id], alpha=0.6)

# Customize plot: adding lines, limits, labels, and legend
for y in [5] + list(range(10, 51, 10)) + [55]:
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
for y in range(15, 46, 10):
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

plt.xlim(-0.5, df['experience_index'].max() + 0.5)
plt.ylim(5, 55)
plt.xticks(df['experience_index'].unique(), df['Category'].unique())
plt.xlabel('Experience category', labelpad=15)
plt.ylabel('PCIst')
plt.title('')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)

# Save and show the figure
plt.savefig('/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/Article_pics/Scatter plots/PCIst_V2.png', dpi=300, bbox_inches='tight')
plt.show()
