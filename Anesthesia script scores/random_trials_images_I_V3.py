import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Getting data 
project_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/'
data_folder = 'derivatives/pipeline_rest'

subject_folders = os.listdir(os.path.join(project_dir, data_folder))
eeg_image_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/images_tms/random trials'
condition = 'rest'

# Mapping file conditions to colors
file_to_color = {
    "task-awake_acq-rest_run-EC": "blue",
    "task-awake_acq-rest_run-EO": "red",
    "task-sed_acq-rest_run-1": "yellow",
    "task-sed_acq-rest_run-2": "green",
    "task-sed_acq-rest_run-3": "grey"
}

# Create a single PDF for all subjects
with PdfPages(os.path.join(eeg_image_dir, "all_subjects_trials.pdf")) as pdf:
    # Loop through subjects
    for subject in subject_folders:
        
        fig, axs = plt.subplots(len(file_to_color), 1, figsize=(15, len(file_to_color) * 5), sharex=True)
        
        # Get all subject data files
        data_files = os.listdir(os.path.join(project_dir, data_folder, subject))

        for idx, (condition_key, color) in enumerate(file_to_color.items()):
            for file in data_files:
                if condition_key in file and "rest" in file:
                    # 1. Load data
                    file_path = os.path.join(project_dir, data_folder, subject, file)
                    epochs = mne.read_epochs(file_path, preload=True)

                    # Randomly choose one trial
                    random_trial_idx = np.random.randint(0, len(epochs))
                    trial_data = epochs[random_trial_idx].get_data()[0]
                    times = epochs.times

                    cz_index = epochs.ch_names.index('Cz')  # Find the index of Cz channel
                    
                    for ch, ch_data in enumerate(trial_data):
                        alpha_value = 0.1 if ch != cz_index else 1  # Lighter for all channels except Cz
                        line_width = 1.5 if ch == cz_index else 1  # Emphasized width for Cz
                        
                        axs[idx].plot(times, ch_data, color=color, alpha=alpha_value, linewidth=line_width)

                    axs[idx].set_title(f"Random Trial from {file}")
                    axs[idx].set_ylabel("Amplitude")
        
        axs[-1].set_xlabel("Time (ms)")
        plt.tight_layout()
        
        # Save the combined figure as a page in the PDF
        pdf.savefig(fig)
        plt.close(fig)