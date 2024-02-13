import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# Directories and paths
project_dir = '/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives'
eeg_folder = 'pipeline_TMS'
subject_folders = os.listdir(os.path.join(project_dir, eeg_folder))
output_dir = os.path.join('/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/images_tms/plots')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define standard parameters
file_extension = '_epo.fif'
condition = 'tms'

# Start a PDF to collect all plots
pdf_filename = os.path.join(output_dir, 'all_trials_plot_with_avg.pdf')

with PdfPages(pdf_filename) as pdf:
    for subject in subject_folders:
        # List all data files for the subject
        data_files = os.listdir(os.path.join(project_dir, eeg_folder, subject))

        for file in data_files:
            if file.endswith(file_extension) and condition in file:
                file_path = os.path.join(project_dir, eeg_folder, subject, file)
                data = mne.read_epochs(file_path, preload=True)
                data.apply_baseline((-0.2, -0.007))

                data_array = data.get_data()

                # Create main figure and subplots
                fig = plt.figure(figsize=(15, 10))
                fig.suptitle(f'File: {file}')

                # Subplot for individual trials
                ax1 = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=3, fig=fig)
                im = ax1.imshow(np.squeeze(data_array[:,data.ch_names.index('Cz'),:]), aspect='auto', origin='lower',
                                extent=[data.times[0], data.times[-1], 0, len(data)],
                                cmap="RdBu_r", clim=[-1e-5, 1e-5])
                fig.colorbar(im, ax=ax1)

                # Calculate the segments
                n_trials = len(data)
                segments = np.array_split(np.arange(n_trials), 3)

                # For each segment, plot the average response
                for idx, segment in enumerate(segments):
                    ax = plt.subplot2grid((3, 4), (idx, 3), fig=fig)
                    
                    # 1. Loop through all channels and plot their average responses in gray
                    for channel in data.ch_names:
                        if channel != 'Cz':
                            avg_response_other = np.squeeze(data_array[segment[0]:segment[-1],data.ch_names.index(channel),:]).mean(axis=0)
                            ax.plot(data.times, avg_response_other, color='lightgray', alpha=0.5)  # alpha for a bit of transparency
                    
                    # 2. Plot the average for 'Cz'
                    avg_response = np.squeeze(data_array[segment[0]:segment[-1],data.ch_names.index('Cz'),:]).mean(axis=0)
                    ax.plot(data.times, avg_response, color="red")
                    
                    ax.axvline(x=0, color='black', linestyle='--')  # vertical line at 0
                    ax.set_title(f'Avg for trials {segment[0]}-{segment[-1]}')
                    ax.set_ylim([-1e-5, 1e-5])
                    
                    # Arrow indication in the main plot
                    mid_trial = (segment[0] + segment[-1]) / 2
                    ax1.annotate('', xy=(data.times[-1], mid_trial),
                                 xytext=(data.times[-1] + 0.1, mid_trial),
                                 arrowprops=dict(arrowstyle="->", lw=1.5))


                # Adjusting plot spacing
                plt.tight_layout()
                plt.subplots_adjust(top=0.90, hspace=0.4)

                # Save the entire figure to the PDF
                pdf.savefig()
                plt.close(fig)
