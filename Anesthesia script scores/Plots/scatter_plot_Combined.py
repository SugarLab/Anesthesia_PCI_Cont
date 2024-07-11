import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

# Set the font globally for all plots
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20

# Load both datasets
df1 = pd.read_excel('/Users/imadjb/Documents/PhD/Anesthesia_conciousness_paper/derivatives/statistics/Python_plotting/lzc_single_V2.xlsx')
df2 = pd.read_excel('/Users/imadjb/Documents/PhD/Anesthesia_conciousness_paper/derivatives/statistics/Python_plotting/PCIst_V2.xlsx')

# Define the desired category order
category_order = ["No experience", "No information", "Experience"]

# Create a categorical type with the specified order for both dataframes
df1['Category'] = pd.Categorical(df1['Category'], categories=category_order, ordered=True)
df2['Category'] = pd.Categorical(df2['Category'], categories=category_order, ordered=True)

# Combine unique Subject_IDs from both datasets
all_unique_ids = pd.concat([df1['Subject_ID'], df2['Subject_ID']]).unique()

# Create a mapping for Subject IDs to "S0, S1, ..., SX"
id_labels = {id: f"S{i}" for i, id in enumerate(sorted(all_unique_ids))}

# Apply the mapping to both dataframes
df1['Subject_Label'] = df1['Subject_ID'].map(id_labels)
df2['Subject_Label'] = df2['Subject_ID'].map(id_labels)

# Create a color palette and map it using the transformed labels
color_palette = sns.color_palette("tab20", len(all_unique_ids))
colors = {id_labels[id]: color_palette[i % 20] for i, id in enumerate(sorted(all_unique_ids))}

# Plotting both datasets with the same color map
fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Plot 1: PCIst
sns.swarmplot(x='Category', y='PCIst', data=df2, hue='Subject_Label', palette=colors, ax=ax[0], size=10, edgecolor='black', linewidth=1)
ax[0].set_ylabel('PCIst')
ax[0].set_ylim(5, 55)  # Set your y-limits as desired
ax[0].set_xlabel('Experience Report Category')  # Update the x-axis label

# Adding horizontal lines
for y in [5] + list(range(10, 51, 10)) + [55]:
    ax[0].axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
for y in range(15, 46, 10):
    ax[0].axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Plot 2: Single Channel LZc
sns.swarmplot(x='Category', y='LZ_single_channel', data=df1, hue='Subject_Label', palette=colors, ax=ax[1], size=10, edgecolor='black', linewidth=1)
ax[1].set_ylabel('Single-channel LZc')
ax[1].set_ylim(0.21, 0.33)  # Set your y-limits as desired
ax[1].set_yticks(np.arange(0.22, 0.33, 0.02))  # Set specific y-ticks
ax[1].set_xlabel('Experience Report Category')  # Update the x-axis label

# Adding horizontal lines
for y in np.arange(0.22, 0.33, 0.01):
    ax[1].axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Set legend with modified subject IDs
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Subject ID', loc='lower center', ncol=5, bbox_to_anchor=(0.51, -0.2))

# Remove individual legends to avoid duplication
ax[0].get_legend().remove()
ax[1].get_legend().remove()

# Save the figure in high resolution
plt.savefig('/Users/imadjb/Documents/PhD/Anesthesia_conciousness_paper/derivatives/Article_pics/Scatter plots/Combined_Plots.png', dpi=300, bbox_inches='tight')
plt.show()
