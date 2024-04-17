import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Specify the path to the single EEG file you want to analyze
project_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/'
data_folder = 'derivatives/pipeline_rest'
subject = 'sub-1074'
eeg_file = 'sub-1074_task-sed_acq-rest_run-1_epo.fif'
eeg_image_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/psd_rest'

# Check if the output directory exists, create if it does not
if not os.path.exists(eeg_image_dir):
    os.makedirs(eeg_image_dir)

# Create a PDF for the output plot
with PdfPages(os.path.join(eeg_image_dir, f"{subject}_PSD_StdDev.pdf")) as pdf:
    # Load the specified EEG file
    file_path = os.path.join(project_dir, data_folder, subject, eeg_file)
    epochs = mne.read_epochs(file_path, preload=True)
    
    # Compute the PSD, focusing on EEG channels
    psd_output = epochs.compute_psd(fmin=1, fmax=40, picks='eeg')
    psds = psd_output.get_data()
    freqs = psd_output.freqs
    
    # Convert PSDs to dB, then compute mean and standard deviation across channels
    psds_dB = 10 * np.log10(psds + np.finfo(float).eps)
    mean_psds_dB = np.mean(psds_dB, axis=1)[0]
    std_psds_dB = np.std(psds_dB, axis=1)[0]

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(freqs, mean_psds_dB, label='Mean PSD')
    ax.fill_between(freqs, mean_psds_dB - std_psds_dB, mean_psds_dB + std_psds_dB, color='gray', alpha=0.5, label='Standard Deviation')

    ax.set_title(" ")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB/Hz)")
    ax.legend()

    plt.tight_layout()
    pdf.savefig(fig)  # Save the figure into a PDF
    plt.close(fig)
