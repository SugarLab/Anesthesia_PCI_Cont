import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from pathlib import Path

# Group 1 Filenames (Confidence Score 1)
group_1_filenames = [
    'sub-1045_task-sed_acq-tms_run-3_epo.pkl',
    'sub-1055_task-sed_acq-tms_run-2_epo.pkl',
    'sub-1057_task-sed_acq-tms_run-2_epo.pkl',
    'sub-1061_task-sed_acq-tms_run-2_epo.pkl',
    'sub-1064_task-sed_acq-tms_run-4_epo.pkl',
    'sub-1067_task-sed_acq-tms_run-4_epo.pkl'
]

# Group 3 Filenames (Confidence Score 3)
group_3_filenames = [
    'sub-1016_task-sed_acq-tms_run-2_epo.pkl',
    'sub-1017_task-sed_acq-tms_run-1_epo.pkl',
    'sub-1022_task-sed_acq-tms_run-1_epo.pkl',
    'sub-1022_task-sed_acq-tms_run-2_epo.pkl',
    'sub-1024_task-sed_acq-tms_run-1_epo.pkl',
    'sub-1046_task-sed_acq-tms_run-1_epo.pkl',
    'sub-1057_task-sed_acq-tms_run-3_epo.pkl',
    'sub-1060_task-sed_acq-tms_run-2_epo.pkl',
    'sub-1061_task-sed_acq-tms_run-1_epo.pkl',
    'sub-1061_task-sed_acq-tms_run-3_epo.pkl',
    'sub-1064_task-sed_acq-tms_run-1_epo.pkl',
    'sub-1064_task-sed_acq-tms_run-2_epo.pkl',
    'sub-1067_task-sed_acq-tms_run-1_epo.pkl',
    'sub-1067_task-sed_acq-tms_run-3_epo.pkl',
    'sub-1071_task-sed_acq-tms_run-2_epo.pkl',
    'sub-1071_task-sed_acq-tms_run-3_epo.pkl',
    'sub-1074_task-sed_acq-tms_run-1_epo.pkl',
    'sub-1074_task-sed_acq-tms_run-2_epo.pkl',
    'sub-1074_task-sed_acq-tms_run-3_epo.pkl'
]


# Function to load and process GMFP data for a list of filenames
def load_gmfp_data(filenames, eeg_folder_path):
    gmfp_data = []

    for filename in filenames:
        subject_id = filename.split('_')[0]
        file_path = eeg_folder_path / subject_id / filename

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        with open(file_path, 'rb') as f:
            itpc_data = pickle.load(f)
        gmfp_data.append(itpc_data['gmfp'])

    return gmfp_data

# Function to calculate mean GMFP
def calculate_mean_gmfp(gmfp_data):
    return np.mean(gmfp_data, axis=0)

# Define paths and parameters
project_dir = Path("/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives")
eeg_folder_path = project_dir / 'pipeline_gmfp'
output_dir = project_dir / 'statistics' / 'gmfp_plots'

# Load GMFP data for both groups
gmfp_data_1 = load_gmfp_data(group_1_filenames, eeg_folder_path)
gmfp_data_3 = load_gmfp_data(group_3_filenames, eeg_folder_path)

# Calculate mean GMFP for both groups
mean_gmfp_1 = calculate_mean_gmfp(gmfp_data_1)
mean_gmfp_3 = calculate_mean_gmfp(gmfp_data_3)

# Plot the data
plt.figure(figsize=(12, 6))


# Plot each individual GMFP in faded color and mean GMFP in thick line for Group 1
for data in gmfp_data_1:
    plt.plot(np.log(data), color='black', alpha=0.1)
plt.plot(np.log(mean_gmfp_1), color='black', linewidth=2, label=' "No experience" Mean GMFP')

# Plot each individual GMFP in faded color and mean GMFP in thick line for Group 3
for data in gmfp_data_3:
    plt.plot(np.log(data), color='red', alpha=0.1)
plt.plot(np.log(mean_gmfp_3), color='red', linewidth=2, label='"Experience" Mean GMFP')

plt.title('Comparison of Logarithm of Mean GMFP Between "No experience" and "Experience"')
plt.xlabel('Time Points')
plt.ylabel('Log(GMFP Value)')
plt.legend()
plt.grid(True)
plt.show()
