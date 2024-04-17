import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set the font to Times New Roman, size 12
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20

# Define the data file path directly
file_path = '/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/pipeline_rest/sub-1057/sub-1057_task-sed_acq-rest_run-2_epo.fif'

# Load data
epochs = mne.read_epochs(file_path, preload=True)

# Randomly choose one trial
random_trial_idx = np.random.randint(0, len(epochs))
trial_data = epochs[random_trial_idx].get_data()[0]
times = epochs.times

# Plotting
plt.figure(figsize=(15, 5))

# Find the index of Cz channel
cz_index = epochs.ch_names.index('Cz')

# Plot all channels in light grey and Cz in black, with Cz on top
for ch, ch_data in enumerate(trial_data):
    color = 'lightgrey'  # Light grey for other channels
    line_width = 0.5  # Standard line width for other channels
    plt.plot(times, ch_data, color=color, linewidth=line_width)  # Adjust y-values by adding 0.00006

# Now plot the Cz channel in black and with a thicker line
plt.plot(times, trial_data[cz_index], color='k', linewidth=2)  # Adjust y-values by adding 0.00006

plt.title("")
plt.ylabel("Amplitude (µV)")
plt.xlabel("Time (ms)")

# Adjust the y-axis to show only -0.00004, 0, and 0.00004
plt.yticks([-0.00002, 0, 0.00002])

# Set the y-axis limits to +/-0.000075
plt.ylim(-0.00004, 0.00004)

# Set x-axis limits to match the data range to remove any blank space
plt.xlim(times[0], times[-1])

plt.tight_layout()

# Show the plot without saving
plt.show()


