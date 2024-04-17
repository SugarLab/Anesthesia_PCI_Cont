import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt

# Define paths
project_dir = Path("/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives")
eeg_folder_path = project_dir / 'pipeline_rest'

# Font settings
font_size = 20  # Set your desired font size here
font_family = 'Times New Roman'  # Set font to Times New Roman

# Apply font settings
plt.rcParams.update({'font.size': font_size, 'font.family': font_family})



filenames = [
    'sub-1045_task-sed_acq-rest_run-3_epo.fif',
    'sub-1055_task-sed_acq-rest_run-1_epo.fif',
    'sub-1055_task-sed_acq-rest_run-2_epo.fif',
    'sub-1057_task-sed_acq-rest_run-2_epo.fif',
    'sub-1061_task-sed_acq-rest_run-2_epo.fif'
]


# filenames = [
#     'sub-1010_task-sed_acq-rest_run-1_epo.fif',
#     'sub-1010_task-sed_acq-rest_run-2_epo.fif',
#     'sub-1010_task-sed_acq-rest_run-3_epo.fif',
#     'sub-1016_task-sed_acq-rest_run-2_epo.fif',
#     'sub-1017_task-sed_acq-rest_run-1_epo.fif',
#     'sub-1017_task-sed_acq-rest_run-2_epo.fif',
#     'sub-1022_task-sed_acq-rest_run-1_epo.fif',
#     'sub-1022_task-sed_acq-rest_run-2_epo.fif',
#     'sub-1024_task-sed_acq-rest_run-1_epo.fif',
#     'sub-1046_task-sed_acq-rest_run-1_epo.fif',
#     'sub-1057_task-sed_acq-rest_run-3_epo.fif',
#     'sub-1060_task-sed_acq-rest_run-2_epo.fif',
#     'sub-1061_task-sed_acq-rest_run-1_epo.fif',
#     'sub-1061_task-sed_acq-rest_run-3_epo.fif',
#     'sub-1064_task-sed_acq-rest_run-1_epo.fif',
#     'sub-1064_task-sed_acq-rest_run-2_epo.fif',
#     'sub-1067_task-sed_acq-rest_run-1_epo.fif',
#     'sub-1067_task-sed_acq-rest_run-3_epo.fif',
#     'sub-1071_task-sed_acq-rest_run-2_epo.fif',
#     'sub-1071_task-sed_acq-rest_run-3_epo.fif',
#     'sub-1074_task-sed_acq-rest_run-1_epo.fif',
#     'sub-1074_task-sed_acq-rest_run-2_epo.fif',
#     'sub-1074_task-sed_acq-rest_run-3_epo.fif'
# ]



# Initialize lists to store PSDs for all subjects
all_mean_psds = []
all_std_psds = []

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
    
    # Compute the PSD for the 'Cz' channel
    psd_output = epochs.compute_psd(fmin=1, fmax=40, picks='Cz')
    psds = psd_output.get_data()  # Shape should be (epochs, channels, frequencies)
    freqs = psd_output.freqs
    
    # Convert PSDs to dB
    psds_dB = 10 * np.log10(psds + np.finfo(float).eps)
    
    # Compute mean and standard deviation across epochs for each frequency
    mean_psds_dB = np.mean(psds_dB, axis=0)[0]  # [0] selects the 'Cz' channel data
    std_psds_dB = np.std(psds_dB, axis=0)[0]

    # Append to the list
    all_mean_psds.append(mean_psds_dB)
    all_std_psds.append(std_psds_dB)

# Convert lists to arrays for mean and std computation
all_mean_psds = np.array(all_mean_psds)
all_std_psds = np.array(all_std_psds)

# Calculate the grand mean and standard deviation across all subjects
grand_mean_psds = np.mean(all_mean_psds, axis=0)
grand_std_psds = np.sqrt(np.mean(all_std_psds**2, axis=0))

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(freqs, grand_mean_psds)
plt.fill_between(freqs, grand_mean_psds - grand_std_psds, grand_mean_psds + grand_std_psds, color='gray', alpha=0.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB/Hz)")
plt.ylim(-120, -70)  # Adjusted y-axis limits
plt.yticks(np.arange(-115, -70, 5))  # Adjust y-axis labels to start at -110, end at -70, increment by 5
plt.xlim(0, 40)  # Set x-axis limits from 0 to 40 Hz

# Save the figure in high resolution
plt.savefig('/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/Article_pics/PSD_rest/no_experience_PSD_average_Cz.png', dpi=300, bbox_inches='tight')

plt.show()