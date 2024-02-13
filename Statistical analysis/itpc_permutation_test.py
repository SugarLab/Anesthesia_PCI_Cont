import numpy as np
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu
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

def aggregate_itpc_data(filenames, eeg_folder_path, data_key, threshold):
    aggregated_data = []
    for filename in filenames:
        subject_id = filename.split('_')[0]
        file_path = eeg_folder_path / subject_id / filename
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
        with open(file_path, 'rb') as f:
            itpc_data = pickle.load(f)
        filtered_data = itpc_data[data_key][(itpc_data[data_key] > threshold) & (itpc_data[data_key] < 0.485)]
        aggregated_data.extend(filtered_data.flatten())
    return aggregated_data

# Define paths and parameters
project_dir = Path("/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives")
eeg_folder_path = project_dir / 'pipeline_itpc_alpha_band'
output_dir = project_dir / 'statistics' / 'itpc_plots'

# Aggregate itpc_d data for both groups with the threshold
threshold_value = 0.015
aggregated_data_1 = aggregate_itpc_data(group_1_filenames, eeg_folder_path, 'itpc_drop', threshold_value)
aggregated_data_3 = aggregate_itpc_data(group_3_filenames, eeg_folder_path, 'itpc_drop', threshold_value)


# Calculate median values for both groups
median_value_1 = np.median(aggregated_data_1)
median_value_3 = np.median(aggregated_data_3)

# Assuming you have phase-locked duration data in two numpy arrays: ce_data and nce_data
ce_data = aggregated_data_1
nce_data = aggregated_data_3

# Combine data
pooled_data = np.concatenate([ce_data, nce_data])

# Number of permutations
n_permutations = 10000
perm_diffs = []

# Compute the original difference
original_diff = np.median(ce_data) - np.median(nce_data)

# Permutation test
for _ in range(n_permutations):
    # Shuffle the pooled data and split into new CE and NCE groups
    np.random.shuffle(pooled_data)
    new_ce = pooled_data[:len(ce_data)]
    new_nce = pooled_data[len(ce_data):]

    # Compute phase-locked durations for permuted data
    # (Implement the same method used for the original data)
    # For example, if it's a mean:
    perm_diff = np.median(new_ce) - np.median(new_nce)

    perm_diffs.append(perm_diff)

# Calculate one-tailed P-value
p_value = np.sum(np.array(perm_diffs) >= original_diff) / n_permutations

print(f"One-tailed P-value: {p_value}")
