import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


# Function to load and plot GMFP data for a single file
def plot_gmfp_data(file_path, pdf):
    with open(file_path, 'rb') as f:
        itpc_data = pickle.load(f)
    
    gmfp_data = itpc_data['gmfp']

    # Plotting
    plt.figure()
    plt.plot(gmfp_data, color='blue')
    plt.title(f'GMFP for {file_path.name}')
    plt.xlabel('Time Points')
    plt.ylabel('GMFP Value')
    plt.grid(True)

    # Save the plot to a PDF page
    pdf.savefig()
    plt.close()

# Define paths and parameters
project_dir = Path("/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives")
eeg_folder_path = project_dir / 'pipeline_gmfp'
output_dir = project_dir / 'statistics' / 'gmfp_plots'

# Group filenames
group_filenames = group_1_filenames + group_3_filenames  # Combine both groups

# PDF file to save all plots
pdf_filename = output_dir / 'individual_gmfp_plots.pdf'
with PdfPages(pdf_filename) as pdf:
    for filename in group_filenames:
        subject_id = filename.split('_')[0]
        file_path = eeg_folder_path / subject_id / filename

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        plot_gmfp_data(file_path, pdf)

print(f"All GMFP plots saved to {pdf_filename}")