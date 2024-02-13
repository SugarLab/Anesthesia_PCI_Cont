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
        filtered_data = itpc_data[data_key][itpc_data[data_key] > threshold]
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


# Determine the common range for the histograms
all_values = aggregated_data_1 + aggregated_data_3
min_value = min(all_values)
max_value = max(all_values)
hist_range = (min_value, max_value)

# Determine the number of bins
num_bins = 20

# Function to find the max count in the histogram
def find_max_count(data, bins, range):
    counts, _ = np.histogram(data, bins=bins, range=range)
    return max(counts)

# Find the maximum count for both groups
max_count_1 = find_max_count(aggregated_data_1, num_bins, hist_range)
max_count_3 = find_max_count(aggregated_data_3, num_bins, hist_range)
common_max_count = max(max_count_1, max_count_3)

# Calculate median values for both groups
median_value_1 = np.median(aggregated_data_1)
median_value_3 = np.median(aggregated_data_3)

# Plot histograms for each group in separate figures
# Histogram for Group 1
plt.figure(figsize=(12, 6))
plt.hist(aggregated_data_1, bins=num_bins, range=hist_range, alpha=0.7, color='blue', label='Group 1')
plt.axvline(x=median_value_1, color='black', linestyle='--', label=f'Median (Group 1: No Experience): {median_value_1:.2f}')
plt.title('Histogram of itpc_d values for Group 1: No Experience')
plt.xlabel('itpc_d Value')
plt.ylabel('Channel Count')
plt.ylim(0, common_max_count)  # Set common y-axis limit
plt.legend()
plt.grid(True)

# Histogram for Group 3
plt.figure(figsize=(12, 6))
plt.hist(aggregated_data_3, bins=num_bins, range=hist_range, alpha=0.7, color='red', label='Group 3')
plt.axvline(x=median_value_3, color='black', linestyle='--', label=f'Median (Group 3: Experience): {median_value_3:.2f}')
plt.title('Histogram of itpc_d values for Group 3: Experience')
plt.xlabel('itpc_d Value')
plt.ylabel('Count')
plt.ylim(0, common_max_count)  # Set common y-axis limit
plt.legend()
plt.grid(True)

# Save the plots to a PDF
pdf_filename = output_dir / 'itpc_histograms.pdf'
with PdfPages(pdf_filename) as pdf:
    pdf.savefig(plt.figure(1))
    pdf.savefig(plt.figure(2))
    plt.close('all')

print(f"Histograms saved to {pdf_filename}")