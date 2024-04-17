import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# getting data 
project_dir = r'/Volumes/IMADS SSD/SSD/'
data_folder = 'derivatives/pipeline_tms'

subject_folders = os.listdir(os.path.join(project_dir, data_folder))
eeg_image_dir = r'/Volumes/IMADS SSD/SSD/derivatives/images_tms'
condition = 'tms'

# loop through subjects
for subject in subject_folders:
    # Create a PDF for the current subject
    with PdfPages(os.path.join(eeg_image_dir, f"{subject}_all_trials.pdf")) as pdf:
        # Get all subject data files
        data_files = os.listdir(os.path.join(project_dir, data_folder, subject))
        # loop through relevant files 
        for file in data_files:
            if condition in file:
                # 1. load data
                file_path = os.path.join(project_dir, data_folder, subject, file)

                if "tms" in file:  # Check if it's an epochs file and has "rest" in its name
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
                    
                    ax.set_title("Random Trial from {}".format(file))
                    ax.set_ylabel("Amplitude")
                    ax.set_xlabel("Time (ms)")
                    plt.tight_layout()
                    
                    # Save the current figure as a page in the PDF
                    pdf.savefig(fig)
                    plt.close(fig)
