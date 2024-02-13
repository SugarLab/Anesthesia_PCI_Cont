import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from ITPC_functions_Imad import (
    compute_pwr_and_phase
)
from matplotlib.gridspec import GridSpec

# Define the base directory
base_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/pipeline_TMS/'

# Loop over all subjects in the base directory
subjects = [subject for subject in os.listdir(base_dir) if subject.startswith('sub-')]

# Define the condition names and an empty dictionary to store the file paths
conditions = [
    "task-awake_acq-rest_run-EC",
    "task-awake_acq-rest_run-EO",
    "task-sed_acq-rest_run-1",
    "task-sed_acq-rest_run-2",
    "task-sed_acq-rest_run-3"
]
file_dict = {condition: [] for condition in conditions}

# Populate the file_dict with paths to the files
for subject in subjects:
    subject_dir = os.path.join(base_dir, subject)
    for file in os.listdir(subject_dir):
        if file.endswith('.fif'):
            for condition in conditions:
                if condition in file:
                    file_path = os.path.join(subject_dir, file)
                    file_dict[condition].append(file_path)

# Now, loop over the conditions and files to execute your processing script
for condition, files in file_dict.items():
    for file in files:
        data = mne.read_epochs(file)

        # Your processing steps
        data_1 = data.get_data().transpose(2,1,0)[:,10:15,:]

        fs_data = int(data.info['sfreq'])

        highest_frequency = 40
        lowest_frequency = 5
        num_wavelets = highest_frequency - lowest_frequency + 1
        b1 = -0.25 
        b2 = -0.05 
        n_straps = 500
        alpha = 0.05

        data_1_pwr_phase = compute_pwr_and_phase(data_1, data, fs=fs_data, highest_frequency=highest_frequency, 
                                                 lowest_frequency=lowest_frequency, num_wavelets=num_wavelets, 
                                                 b1=b1, b2=b2, n_straps=n_straps, alpha=alpha)

        # Plot results
        plt.figure()

        # Plot the averaged data
        plt.subplot(311)
        mean_data_length = data_1.mean(axis=2).shape[0]
        plt.plot(data_1.mean(axis=2))
        plt.xticks(np.linspace(0, mean_data_length, 4), np.linspace(-250, 500, 4))
        plt.title(condition)

        # Display the first image
        plt.subplot(312)
        img1_length = data_1_pwr_phase[0][:,:,3].shape[1]
        plt.imshow(np.squeeze(data_1_pwr_phase[0][:,:,3]))
        plt.xticks(np.linspace(0, img1_length, 4), np.linspace(-250, 500, 4))

        # Display the second image
        plt.subplot(313)
        img2_length = data_1_pwr_phase[1][:,:,3].shape[1]
        plt.imshow(np.squeeze(data_1_pwr_phase[1][:,:,3]))
        plt.xticks(np.linspace(0, img2_length, 4), np.linspace(-250, 500, 4))
        
        # Optionally, you can save each figure with a unique name
        plt.savefig(os.path.join(subject_dir, f"{condition}_result_plot.png"))
        plt.close()
