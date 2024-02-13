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


# Function to load data from files
def load_data(filenames, eeg_folder_path, data_key):
    data_group = []
    for filename in filenames:
        subject_id = filename.split('_')[0]
        file_path = eeg_folder_path / subject_id / filename
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
        with open(file_path, 'rb') as f:
            itpc_data = pickle.load(f)
        data_group.append(itpc_data[data_key])
    return data_group

# Function to perform Mann-Whitney U test and plot p-values
def plot_p_values(gmfp_data_1, gmfp_data_3, title, pdf):
    num_time_points = len(gmfp_data_1[0])
    significance_level = 0.05
    p_values = []

    # Mann-Whitney U test for each time point
    for i in range(num_time_points):
        gmfp_at_time_1 = [gmfp[i] for gmfp in gmfp_data_1]
        gmfp_at_time_3 = [gmfp[i] for gmfp in gmfp_data_3]
        _, p_value = mannwhitneyu(gmfp_at_time_1, gmfp_at_time_3, alternative='two-sided')
        p_values.append(p_value)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(range(num_time_points), p_values, label='P-Values', color='blue')
    plt.axhline(y=significance_level, color='green', linestyle='--', label='Significance Threshold (0.05)')
    plt.axvline(x=250, color='red', linestyle='--', label='Time Point 250')
    plt.xlabel('Time Points')
    plt.ylabel('P-Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()

# Define paths and parameters
project_dir = Path("/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives")
eeg_folder_path = project_dir / 'pipeline_gmfp'
output_dir = project_dir / 'statistics' / 'gmfp_plots'


# PDF file to save all plots
pdf_filename = output_dir / 'gmfp_p_value_plots.pdf'
with PdfPages(pdf_filename) as pdf:
    for data_key in ['gmfp', 'first_n_gmfp', 'last_n_gmfp']:
        gmfp_data_1 = load_data(group_1_filenames, eeg_folder_path, data_key)
        gmfp_data_3 = load_data(group_3_filenames, eeg_folder_path, data_key)
        plot_p_values(gmfp_data_1, gmfp_data_3, f'P-Values Over Time Points for {data_key}', pdf)

print(f"All GMFP plots saved to {pdf_filename}")
