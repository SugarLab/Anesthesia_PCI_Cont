import numpy as np
import mne
import os
import matplotlib.pyplot as plt

# Specify the paths and file you want to plot
project_dir = '/Volumes/IMADS SSD/Anesthesia_conciousness_paper/'
data_folder = 'derivatives/pipeline_tms'
eeg_image_dir = '/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/Article_pics/Evoked_TMS'

# Specify the subject and file you want to analyze
subject = 'sub-1057'
file_name = 'sub-1057_task-sed_acq-tms_run-2_epo.fif'

# Font settings
plt.rcParams.update({'font.size': 20, 'font.family': 'Times New Roman'})

file_path = os.path.join(project_dir, data_folder, subject, file_name)
epochs = mne.read_epochs(file_path, preload=True)

# Compute average across trials for each channel
average_data = epochs.average().data  # Shape: (n_channels, n_times)
times = epochs.times

fig, ax = plt.subplots(figsize=(15, 5))

# Plot all channels in grey
for ch_data in average_data:
    ax.plot(times, ch_data, color='grey', alpha=0.5)

# Emphasize the Cz channel in black
cz_index = epochs.ch_names.index('Cz')
ax.plot(times, average_data[cz_index], color='black', linewidth=1.5)

ax.set_title("")
ax.set_ylabel("Amplitude (µV)")
ax.set_xlabel("Time (s)")

# Set y-axis limits and labels
ax.set_ylim(-0.000025, 0.000025)
ax.set_yticks(np.linspace(-0.00002, 0.00002, 5))

# Set x-axis limits to the range of your data to remove white space
ax.set_xlim(times[0], times[-1])

plt.tight_layout()

# Save the figure as a high-resolution PNG image
image_path = os.path.join(eeg_image_dir, f"{subject}_{file_name.replace('.fif', '')}.png")
plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.show()
