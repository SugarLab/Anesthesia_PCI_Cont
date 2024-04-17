import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# Directories and paths
project_dir = '/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives'
eeg_folder = 'pipeline_TMS'
subject = 'sub-1057'  # Specify the subject folder here
file_name = 'sub-1057_task-sed_acq-tms_run-2_epo.fif'  # Specify the exact file to plot

output_dir = os.path.join(project_dir, 'images_tms', 'plots')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Start a PDF to collect the plot
pdf_filename = os.path.join(output_dir, f'{subject}_{file_name.replace(".fif", "")}_plot_with_avg.pdf')

with PdfPages(pdf_filename) as pdf:
    file_path = os.path.join(project_dir, eeg_folder, subject, file_name)
    data = mne.read_epochs(file_path, preload=True)
    data.apply_baseline((-0.2, -0.007))

    data_array = data.get_data()

    # Create main figure and subplots with adjusted size
    fig = plt.figure(figsize=(18, 10))  # Increase figure width
    fig.suptitle(f'File: {file_name}')

    # Subplot for individual trials
    ax1 = plt.subplot2grid((3, 5), (0, 0), rowspan=3, colspan=4, fig=fig)  # Adjust grid for main plot
    im = ax1.imshow(np.squeeze(data_array[:, data.ch_names.index('Cz'), :]), aspect='auto', origin='lower',
                    extent=[data.times[0], data.times[-1], 0, len(data)],
                    cmap="RdBu_r", clim=[-1e-5, 1e-5])
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Trials')
    fig.colorbar(im, ax=ax1)

    # Calculate the segments
    n_trials = len(data)
    segments = np.array_split(np.arange(n_trials), 3)

    # Determine y-limits
    all_responses = []
    for segment in segments:
        for channel in data.ch_names:
            avg_response = np.squeeze(data_array[segment[0]:segment[-1], data.ch_names.index(channel), :]).mean(axis=0)
            all_responses.append(avg_response)

    y_min, y_max = np.min(all_responses), np.max(all_responses)

    # Plotting with consistent y-limits and highlighted 'Cz'
    for idx, segment in enumerate(segments):
        ax = plt.subplot2grid((3, 5), (idx, 4), fig=fig)  # Adjust grid for right side plots

        # Plot all other channels first
        for channel in data.ch_names:
            if channel != 'Cz':
                avg_response_other = np.squeeze(data_array[segment[0]:segment[-1], data.ch_names.index(channel), :]).mean(axis=0)
                ax.plot(data.times, avg_response_other, color='lightgray', alpha=0.5)

        # Now plot 'Cz' on top
        avg_response_cz = np.squeeze(data_array[segment[0]:segment[-1], data.ch_names.index('Cz'), :]).mean(axis=0)
        ax.plot(data.times, avg_response_cz, color="black", alpha=1.0)

        ax.axvline(x=0, color='black', linestyle='--')
        ax.set_title(f'Avg for trials {segment[0]}-{segment[-1]}')
        ax.set_ylim([y_min, y_max])
        if idx == len(segments) - 1:  # Only set x-axis label for the last subplot
            ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')

    # Adjusting plot spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.4, left=0.1)

    # Save the entire figure to the PDF
    pdf.savefig()
    plt.close(fig)
