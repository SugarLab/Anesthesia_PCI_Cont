import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Getting data 
project_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/'
data_folder = 'derivatives/pipeline_rest'
subject_folders = os.listdir(os.path.join(project_dir, data_folder))
eeg_image_dir = r'//Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/psd_rest'

# Mapping file conditions to colors
file_to_color = {
    "task-awake_acq-rest_run-EC": "blue",
    "task-awake_acq-rest_run-EO": "red",
    "task-sed_acq-rest_run-1": "yellow",
    "task-sed_acq-rest_run-2": "green",
    "task-sed_acq-rest_run-3": "grey"
}

# Storage for average PSDs for each condition and subject
condition_avg_data = {key: [] for key in file_to_color.keys()}

# Create a PDF for all subjects
with PdfPages(os.path.join(eeg_image_dir, "all_subjects_PSD.pdf")) as pdf:
    # Loop through subjects
    for subject in subject_folders:
        data_files = os.listdir(os.path.join(project_dir, data_folder, subject))

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.set_xlim(1, 40)  # Setting a fixed x-axis range from 1 to 40 Hz

        # Loop through relevant files 
        for condition_key, color in file_to_color.items():
            for file in data_files:
                if condition_key in file and file.endswith('.fif'):
                    # 1. Load data
                    file_path = os.path.join(project_dir, data_folder, subject, file)
                    epochs = mne.read_epochs(file_path, preload=True)
                    
                    # Compute the PSD
                    psd_output = epochs.compute_psd(fmin=1, fmax=40)  # Limit fmax to 40
                    psds = psd_output.get_data()
                    epsilon = 1e-10  # a small value to prevent log(0)
                    psds_dB = 10 * np.log10(psds + epsilon)
                    freqs = psd_output.freqs  # Extract frequencies for plotting
                    
                    # Calculate the mean PSD across all channels
                    avg_psds_dB = np.mean(psds_dB, axis=1)[0]
                    condition_avg_data[condition_key].append(avg_psds_dB)

                    ax.plot(freqs, avg_psds_dB, label=condition_key, color=color)

        ax.set_title(f"Averaged PSD for {subject}")
        ax.set_ylabel("Power Spectral Density (dB)")
        ax.set_xlabel("Frequency (Hz)")
        ax.legend(loc='upper right')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Check the lengths of data stored for each condition
    print("Number of subjects with data for each condition:")
    for condition_key in file_to_color:
        print(f"{condition_key}: {len(condition_avg_data[condition_key])}")


    # Plot grand average for each condition
    fig, ax = plt.subplots(figsize=(15, 5))
    for condition_key, color in file_to_color.items():
        grand_avg = np.mean(condition_avg_data[condition_key], axis=0)
        ax.plot(freqs, grand_avg, label=condition_key, color=color)
    
    ax.set_title("Grand Average PSD across Subjects")
    ax.set_ylabel("Power Spectral Density (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
