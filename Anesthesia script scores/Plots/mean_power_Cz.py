import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt

# Define paths
project_dir = Path("/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives")
eeg_folder_path = project_dir / 'pipeline_tms'

# Font settings
plt.rcParams.update({'font.size': 20, 'font.family': 'Times New Roman'})

# filenames = [
#     'sub-1045_task-sed_acq-tms_run-3_epo.fif',
#     'sub-1055_task-sed_acq-tms_run-2_epo.fif',
#     'sub-1057_task-sed_acq-tms_run-2_epo.fif',
#     'sub-1061_task-sed_acq-tms_run-2_epo.fif',
#     'sub-1064_task-sed_acq-tms_run-4_epo.fif',
#     'sub-1067_task-sed_acq-tms_run-4_epo.fif'
# ]

filenames = [
    'sub-1016_task-sed_acq-tms_run-2_epo.fif',
    'sub-1017_task-sed_acq-tms_run-1_epo.fif',
    'sub-1022_task-sed_acq-tms_run-1_epo.fif',
    'sub-1022_task-sed_acq-tms_run-2_epo.fif',
    'sub-1024_task-sed_acq-tms_run-1_epo.fif',
    'sub-1046_task-sed_acq-tms_run-1_epo.fif',
    'sub-1057_task-sed_acq-tms_run-3_epo.fif',
    'sub-1060_task-sed_acq-tms_run-2_epo.fif',
    'sub-1061_task-sed_acq-tms_run-1_epo.fif',
    'sub-1061_task-sed_acq-tms_run-3_epo.fif',
    'sub-1064_task-sed_acq-tms_run-1_epo.fif',
    'sub-1064_task-sed_acq-tms_run-2_epo.fif',
    'sub-1067_task-sed_acq-tms_run-1_epo.fif',
    'sub-1067_task-sed_acq-tms_run-3_epo.fif',
    'sub-1071_task-sed_acq-tms_run-2_epo.fif',
    'sub-1071_task-sed_acq-tms_run-3_epo.fif',
    'sub-1074_task-sed_acq-tms_run-1_epo.fif',
    'sub-1074_task-sed_acq-tms_run-2_epo.fif',
    'sub-1074_task-sed_acq-tms_run-3_epo.fif'
]

# Initialize a list to store the mean absolute data for the 'Cz' channel across files
abs_cz_means = []

for filename in filenames:
    # Construct file path
    subject_id = filename.split('_')[0]
    file_path = eeg_folder_path / subject_id / filename
    
    # Check if the file exists
    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    # Load the epoch data
    epochs = mne.read_epochs(file_path, preload=True)
    
    # Extract data for 'Cz' channel
    cz_index = epochs.ch_names.index('Cz')
    cz_data = epochs.get_data()[:, cz_index, :]  # Shape: (n_epochs, n_times)
    
    # Compute the mean across epochs for 'Cz' and then take the absolute value
    mean_cz = np.mean(cz_data, axis=0)
    abs_mean_cz = np.abs(mean_cz)
    
    # Append the absolute mean to the list
    abs_cz_means.append(abs_mean_cz)

# No conversion to numpy array for mean calculation (keeping the list structure)
mean_abs_cz_across_files = np.mean(abs_cz_means, axis=0)
std_abs_cz_across_files = np.std(abs_cz_means, axis=0)


# Ensure the output directory exists
output_dir = project_dir / 'Article_pics'
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(15, 5))
times = epochs.times
plt.plot(times, mean_abs_cz_across_files * 1e6)  # Scale the data by 1e4
plt.fill_between(times, (mean_abs_cz_across_files - std_abs_cz_across_files) * 1e6, (mean_abs_cz_across_files + std_abs_cz_across_files) * 1e6, color='gray', alpha=0.5)  # Scale the data by 1e4
plt.xlabel("Time (s)")
plt.ylabel("Absolute Mean Amplitude (×1e-6 µV)")  # Adjust label to reflect new unit
plt.ylim(-1, 12)  # Adjust y-axis limits to reflect the scaling
plt.xlim(-0.25, 0.5)  # Set x-axis limits from -0.2 to 0.5 seconds

# Save the figure in high resolution
output_path = output_dir / 'Cz_mean_field_exp.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()
