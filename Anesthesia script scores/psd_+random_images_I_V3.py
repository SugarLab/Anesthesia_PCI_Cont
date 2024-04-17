import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# Getting data 
project_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/'
data_folder = 'derivatives/pipeline_rest'
subject_folders = os.listdir(os.path.join(project_dir, data_folder))
eeg_image_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/images_rest/'

file_to_color = {
    "task-awake_acq-rest_run-EC": "blue",
    "task-awake_acq-rest_run-EO": "red",
    "task-sed_acq-rest_run-1": "yellow",
    "task-sed_acq-rest_run-2": "green",
    "task-sed_acq-rest_run-3": "grey"
}

condition_avg_data = {key: [] for key in file_to_color.keys()}

with PdfPages(os.path.join(eeg_image_dir, "all_subjects_merged.pdf")) as pdf:
    for subject in subject_folders:
        fig = plt.figure(figsize=(20, len(file_to_color) * 5))
        
        # Adjusting width_ratios to make PSD plots bigger
        gs = GridSpec(len(file_to_color), 2, width_ratios=[3, 1])

        data_files = os.listdir(os.path.join(project_dir, data_folder, subject))

        # --- LEFT: PSDs ---
        ax_psd = fig.add_subplot(gs[:, 0])
        ax_psd.set_xlim(1, 40)

        for condition_key, color in file_to_color.items():
            for file in data_files:
                if condition_key in file and file.endswith('.fif'):
                    file_path = os.path.join(project_dir, data_folder, subject, file)
                    epochs = mne.read_epochs(file_path, preload=True)
                    
                    psd_output = epochs.compute_psd(fmin=1, fmax=40)
                    psds = psd_output.get_data()
                    epsilon = 1e-10
                    psds_dB = 10 * np.log10(psds + epsilon)
                    freqs = psd_output.freqs
                    
                    avg_psds_dB = np.mean(psds_dB, axis=1)[0]
                    condition_avg_data[condition_key].append(avg_psds_dB)
                    ax_psd.plot(freqs, avg_psds_dB, label=condition_key, color=color)

        ax_psd.set_title(f"Averaged PSD for {subject}")
        ax_psd.set_ylabel("Power Spectral Density (dB)")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.legend(loc='upper right')

        # --- RIGHT: Random Trials ---
        for idx, (condition_key, color) in enumerate(file_to_color.items()):
            ax_trial = fig.add_subplot(gs[idx, 1])
            for file in data_files:
                if condition_key in file and "rest" in file:
                    file_path = os.path.join(project_dir, data_folder, subject, file)
                    epochs = mne.read_epochs(file_path, preload=True)

                    random_trial_idx = np.random.randint(0, len(epochs))
                    trial_data = epochs[random_trial_idx].get_data()[0]
                    times = epochs.times

                    cz_index = epochs.ch_names.index('Cz') 
                    
                    for ch, ch_data in enumerate(trial_data):
                        alpha_value = 0.1 if ch != cz_index else 1 
                        line_width = 1.5 if ch == cz_index else 1 
                        
                        ax_trial.plot(times, ch_data, color=color, alpha=alpha_value, linewidth=line_width)

                    ax_trial.set_title(f"Random Trial from {file}")
                    ax_trial.set_ylabel("Amplitude")

            ax_trial.set_xlabel("Time (ms)")
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Add a grand average page at the end
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

