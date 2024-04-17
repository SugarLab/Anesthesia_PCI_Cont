import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import mne
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Define paths and parameters
project_dir = Path("/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives")
eeg_folder = 'pipeline_itpc'
output_dir = project_dir / 'statistics' / 'itpc_topo'
file_extension = '.pkl'
condition = 'tms'

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize DataFrame
df_results = pd.DataFrame()

raw = mne.io.read_raw_brainvision("/Volumes/IMADS SSD/Anesthesia_conciousness_paper/EEG/sub-1016/sub-1016_task-awake_acq-rest_run-EC.vhdr", preload=True)
raw.drop_channels(['VEOG','HEOG','EMG'])
raw.set_montage('standard_1020')

# Create a PDF to save plots
pdf_path = output_dir / 'all_subjects_topoplots.pdf'
with PdfPages(pdf_path) as pdf:
    # Iterate over subject folders
    for subject in os.listdir(project_dir / eeg_folder):

        subject_path = project_dir / eeg_folder / subject

        # Check if it's a folder
        if subject_path.is_dir():

            print(f"Processing: {subject_path}")

            # Initialize a dictionary to store the data for the subject
            subject_data = {}

            # loop through files in the subject folder 
            for file_name in os.listdir(subject_path):

                if file_name.endswith(file_extension) and condition in file_name:

                    print(f"Found file: {file_name}")

                    # Load data
                    with open(subject_path / file_name, 'rb') as f:
                        itpc_data = pickle.load(f)

                    # Retrieve the ITPC value
                    itpc_value_first_15 = itpc_data['first_n_itpc_drop'][:62]
                    itpc_value = itpc_data['itpc_drop'][:62]
                    itpc_value_last_15 = itpc_data['last_n_itpc_drop'][:62]

                    # Create a 3x1 subplot for the three topomaps
                    fig, axarr = plt.subplots(3, 1, figsize=(8, 12))
                    mne.viz.plot_topomap(itpc_value_first_15, raw.info, axes=axarr[0], vlim=(0, 0.5), show=False)
                    mne.viz.plot_topomap(itpc_value, raw.info, axes=axarr[1], vlim=(0, 0.5), show=False)
                    mne.viz.plot_topomap(itpc_value_last_15, raw.info, axes=axarr[2], vlim=(0, 0.5), show=False)
                    axarr[0].set_title(f'{subject} - itpc_value_first_15')
                    axarr[1].set_title(f'{subject} - itpc_value')
                    axarr[2].set_title(f'{subject} - itpc_value_last_15')
                    
                    # Save the figure to the PDF
                    pdf.savefig(fig)
                    plt.close(fig)
