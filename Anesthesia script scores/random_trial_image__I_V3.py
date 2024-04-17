import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Specify the paths and file you want to plot
project_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/'
data_folder = 'derivatives/pipeline_tms'
eeg_image_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/images_tms'

# Specify the subject and file you want to analyze
subject = 'sub-1055'  # Update this to your specific subject's folder name
file_name = 'sub-1055_task-sed_acq-tms_run-2_epo.fif'  # Update this to the specific file you want to plot

# Create a PDF for the plot
pdf_path = os.path.join(eeg_image_dir, f"{subject}_{file_name.replace('.fif', '')}.pdf")
with PdfPages(pdf_path) as pdf:
    file_path = os.path.join(project_dir, data_folder, subject, file_name)
    epochs = mne.read_epochs(file_path, preload=True)
    
    # Randomly choose one trial
    random_trial_idx = np.random.randint(0, len(epochs))
    trial_data = epochs[random_trial_idx].get_data()[0]
    times = epochs.times
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot all channels in light grey
    for ch_data in trial_data:
        ax.plot(times, ch_data, color='lightgrey')
    
    # Emphasize the Cz channel
    emphasized_channel_idx = epochs.ch_names.index('Cz')  # Find the index of Cz channel
    ax.plot(times, trial_data[emphasized_channel_idx], color='black', linewidth=1.5)
    
    ax.set_title(f"")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_xlabel("Time (ms)")
    plt.tight_layout()
    
    # Save the current figure as a page in the PDF
    pdf.savefig(fig)
    plt.close(fig)
