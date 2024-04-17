import mne
import time
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import simps
import os
from autoreject import AutoReject
from PCIst import *
import mne_icalabel as il
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
import io




### do we need these functions?

# Define some functions
        
def plot_response(signal, argument):
    """plot response to check what happened with the data"""
    if "time" in argument:
        signal.plot(duration=10, remove_dc=False)
    if "psd" in argument:
        signal.plot_psd(fmin=0, fmax=80)
    if "butter" in argument:
        signal.plot(butterfly=True, color='#00000044', bad_color='r')
    if "ica" in argument:
        signal.plot_components()


def detect_bad_ch(eeg):
    """plots each channel so user can decide whether good (mouse click) or bad (enter / space)"""
    good_ch, bad_ch = [], []
    intvl = eeg.__len__() // 20
    if type(eeg) is mne.epochs.EpochsArray:
        # Benny's way is way too slow.... and a bit ugly...         
        # Let's try it MNE style
        n_chan = eeg.ch_names.__len__()
        n_disp = 8
        for i in range(0,n_chan,n_disp):
            # Choose 4 channels at a time
            cur_picks = eeg.ch_names[i:(i + n_disp)]
            fig = eeg.plot(picks=cur_picks, title='Click channel names to reject, click on epochs to reject epoch; Use mouse to move around figure, press any key to advance')

            # Wait until keyboard is pressed
            while not plt.waitforbuttonpress():            
                print('Inspecting channels..')

            plt.close(fig)
        return eeg
    else:
        for ch in eeg.ch_names:
            """loop over each channel and plot to decide if bad"""
            time_data = eeg[eeg.ch_names.index(ch)][0][0]
            df = pd.DataFrame()
            for i in range(20):
                df_window = pd.DataFrame(time_data[i * intvl:(i + 1) * intvl])
                df_window += (i + 1) * 0.0001
                df = pd.concat((df, df_window), axis=1)

            df *= 1000  # just for plotting
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(f"{ch}: mouse click for keep (good), any other key for remove (bad)")
            ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1)
            ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=3)
            ax1.psd(time_data, 5000, 5000)
            ax1.set_xlim([0, 55])
            ax2.plot(df, 'b')
            plt.show()

            if not plt.waitforbuttonpress():
                good_ch.append(ch)
                plt.close(fig)
            else:
                bad_ch.append(ch)
                plt.close(fig)

        return good_ch, bad_ch

def detect_bad_ic(ica_data, data_orig):
    """plots each independent component so user can decide whether good (mouse click) or bad (enter / space)"""
    good_ic, bad_ic = [], []
    bad_list = []
    "!!Change back to full range!!"
    for c in range((ica_data.get_components().shape[1])): 
        """loop over each channel and plot to decide if bad"""
        ica_data.plot_properties(inst=data_orig, picks=c)

        if not plt.waitforbuttonpress():
            good_ic.append(c)
            plt.close()
        else:
            bad_ic.append(c)
            plt.close()

    #[bad_list.append(ica_data.ch_names.index(ci)) for ci in bad_ic]
    return bad_ic


## ICA cleaning
def run_ica_cleaning(epochs, n_components=15, random_state=100):

    # load data into epochs object
    epochs.load_data()
    
    # define ica object
    ica = mne.preprocessing.ICA(
        n_components=n_components, 
        random_state=random_state,
        method='infomax', 
        fit_params=dict(extended=True)
    )

    # run ic_label to identify/predict source of ICs
    exclude_idx = []
    ica.fit(epochs)
    ic_labels = il.label_components(epochs, ica, method="iclabel")
    
    # mark IC's that are predicted to be "eye blinks" or "muscle artefacts" 
    # with more than 50% confidence
    exclude_idx = [
        i for (i,label),pred in zip(enumerate(ic_labels['labels']),ic_labels['y_pred_proba'])
        if ((label=='eye blink' or label=='muscle artifact' or label=='channel noise') and pred>0.75)
    ]

    # reconstruct epoch data without non-brain components
    reconst_raw = epochs.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)
    
    return reconst_raw, ica, ic_labels, exclude_idx


def epoch(data, sfreq=1000, tepoch=5, time=5, tmin=-2.5, tmax=2.5):
    # Set parameters
    time = time * 60 # time in sec
    nepochs = int((time / tepoch)) # Number of epochs
    sp = sfreq * tepoch # Samples per epoch = frequency times 5 sec.
    fiveMin =  time * sfreq # 5 minutes in sample points
    points = np.arange(0, fiveMin, sp).reshape(nepochs,1) # Indecies for event points
    dummy = np.ones((nepochs, 1), dtype=int)
    
    # Make event structure
    events = np.concatenate((points,dummy,dummy), 1)
    # Epoch data
    epoched_Data = mne.Epochs(data, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, reject=None)
    
    return epoched_Data

def on_click(event):
    global points
    if event.dblclick:
        points.append(event.xdata)
        if len(points) == 1:
            plt.disconnect(cid)
            plt.close()


# Get all subject folders
project_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/'
eeg_folder = 'EEG'
subject_folders = os.listdir(os.path.join(project_dir,eeg_folder))

# define output folder
output_dir = os.path.join(project_dir,r'derivatives/pipeline_TMS')

# Define standard parameters
file_extension = 'vhdr'
condition = 'tms'

# Initiate results table OUTSIDE the main loop
columns = [
    'subject', 'raw_data_file', 'n_components_removed', 'pulse_start', 'pulse_stop',
    'pct_segments_interpolated', 'pct_segments_rejected', 'n_segments',
    'pct_trials_rejected', 'n_trials', 'n_bad_channels', 'pci_st'
]
results_table = pd.DataFrame(columns=columns)

# loop through subjects
for subject in subject_folders:
    
    # Get all subject data files
    data_files = os.listdir(os.path.join(project_dir,eeg_folder,subject))
    
    # loop through relevant files 
    for file in data_files:
        
        if file.endswith(file_extension) and condition in file:
            
            # create output folder for subject
            outpath = os.path.join(output_dir,subject)
            try: 
                os.mkdir(outpath)
                print('Path created')
            except:
                print("Path exists")
               
            # Create a temporary DataFrame for the current subject
            subject_results = pd.DataFrame(index=[subject], columns=columns)

            # update results dictionary
            subject_results.loc[subject, 'subject'] = subject
        
            # 1. load data (vhdr file)
            file_path = os.path.join(project_dir,eeg_folder,subject,file)
            data = mne.io.read_raw_brainvision(file_path)
            data.load_data()
            
            # crop data from 1s before first pulse, to 1s after last
            #find all pulses 
            pulse_times = [val['onset'] for val in data.annotations if val['description']=='Response/R128']
            data.crop(tmin=pulse_times[0]-1, tmax=pulse_times[-1]+1)
            
            # update results dictionary
            subject_results.loc[subject, 'raw_data_file'] = file
            
            # 1.1. channel info (remove EMG and set type for EOG channels)
            if data.info['ch_names'][-1] == 'EMG':
                data.drop_channels('EMG')
                data.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog', 'EMG': 'emg'})
            else:
                data.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
            data.set_montage('standard_1005')
            
            # 2. remove pulse artifact
            events = mne.events_from_annotations(data)
            epochs = mne.Epochs(
                data, events[0], event_id=events[1]['Response/R128'], tmin=-0.25, tmax=0.5, preload=True, baseline=(-0.25,-0.02)
            ) # Baseline applied
            
            # create raw butterfly plot
            butterfly = epochs.get_data()[:,:62,:].mean(axis=0)
            
            # Try removing pulse artefact, until satisfied
            satisfied = False
            while not satisfied:            
                # Initialize an empty list to store selected points
                points = []
                
                # plot butterfly
                plt.plot(epochs.times,np.transpose(butterfly))
                plt.ylim([-0.00005,0.00005])
                plt.xlim([-0.005, 0.05])
                plt.title('Please double-click on the x-position where the pulse artifact is done')
                
                # Connect the on_click function to the figure
                cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)
                
                # Show the interactive plot
                plt.show()
                
                while len(points) < 1:
                    plt.pause(0.1)
                
                # Retrieve the selected x-values from the points list
                pulse_start = -0.002
                pulse_end = points[0]
                
                # Remove pulse artefact in test data
                data_no_pulse = data.copy()
                mne.preprocessing.fix_stim_artifact(
                    data_no_pulse, events=events[0], event_id=events[1]['Response/R128'], tmin=pulse_start, tmax=pulse_end, mode='linear', picks=['eeg','eog']
                )
                
                # epoch and plot test data
                test_epochs = mne.Epochs(
                    data_no_pulse, events[0], event_id=events[1]['Response/R128'], tmin=-0.25, tmax=0.5, preload=True, baseline=(-0.25,-0.02)
                ) # Baseline applied
                test_epochs.set_eeg_reference()
                test_butterfly = test_epochs.get_data()[:,:62,:].mean(axis=0)
                
                # plot butterfly with suggested pulse artefact removed
                plt.plot(test_epochs.times,np.transpose(test_butterfly))
                plt.ylim([-0.00002,0.00002])
                plt.xlim([-0.05, 0.15])
                plt.title(f"mouse click if satisfied with removal. otherwise press space to try again")
                plt.show()
    
                if not plt.waitforbuttonpress():
                    satisfied = True
                    plt.close()
                else:
                    plt.close()

            # Manually reject bad channels from butterfly
            # create dummy raw data for rejection
            avg_data = test_epochs.get_data().mean(0) 
            dummy_data = mne.io.RawArray(avg_data, test_epochs.info, first_samp=0, copy='auto', verbose=None)
            
            # plot dummy data and select bad channels
            dummy_data.plot(n_channels=62, scalings=2e-6, clipping=8, block=True)
            
            # catch bad channels and save as 'bads' in data_no_pulse
            data_no_pulse.info['bads'] = dummy_data.info['bads']
            
            # interpolate bads
            data_no_pulse.interpolate_bads()
            
            # update results dictionary
            subject_results.loc[subject, 'pulse_start'] = pulse_start
            subject_results.loc[subject, 'pulse_stop'] = pulse_end
            subject_results.loc[subject, 'n_bad_channels'] = len(dummy_data.info['bads'])
    
            
            # 2. resample (with low-pass filter!) and filter data
            new_sampling = 1000.
            hp_freq = 0.5
            lp_freq = 40
            
            # resample
            data_no_pulse.resample(new_sampling, npad='auto')
            
            # Band-pass filter
            data_no_pulse.filter(l_freq=hp_freq, h_freq=lp_freq, method='fir', picks=['eeg','eog'])
            
            # 3. regress out eye artefacts
            # Add reference channel for regression to work
            eog_data = mne.add_reference_channels(data_no_pulse,['ref'],copy=True)
            eog_data,_ = mne.set_eeg_reference(eog_data,['ref'])
            # fit regression
            weights = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog').fit(eog_data)
            # apply regression to correct eye artefacts
            eog_data_clean = weights.apply(eog_data, copy=True)
            eog_data_clean.drop_channels(['ref'])
            
            # 4.2 Apply autoreject
            # Epoch data for cleaning
            events = mne.events_from_annotations(data_no_pulse)
            clean_epochs = mne.Epochs(
                data_no_pulse, events[0], event_id=events[1]['Response/R128'], tmin=-0.25, tmax=0.5, preload=True, baseline=(-0.2,-0.02)
            ) 
            
            ar = AutoReject(verbose='tqdm',picks='eeg')
            ar.fit(clean_epochs)  # fit on a few epochs to save time
            epochs_ar, reject_log = ar.transform(clean_epochs, return_log=True)
            
            # Get stats for bad trials, channels, and interpolated segments
            rejected_segments = np.sum(np.array(
                [
                    [1 if i == 1. else 0 for i in row] 
                    for row in reject_log.labels
                ]
            ))
            interpolated_segments = np.sum(np.array(
                [
                    [1 if i == 2. else 0 for i in row] 
                    for row in reject_log.labels
                ]
            ))
            total_segments = np.count_nonzero(~np.isnan(reject_log.labels))
            
            # store stats in data dictionary
            subject_results.loc[subject, 'pct_segments_interpolated'] = interpolated_segments/total_segments
            subject_results.loc[subject, 'pct_segments_rejected'] = rejected_segments/total_segments
            subject_results.loc[subject, 'n_segments'] = total_segments
            subject_results.loc[subject, 'pct_trials_rejected'] = sum(reject_log.bad_epochs)/len(reject_log.bad_epochs)
            subject_results.loc[subject, 'n_trials'] = len(reject_log.bad_epochs)
            
            # 5. ICA
            
            # 5.1 Rereference to average because otherwise ICA is sad
            epochs_ar.set_eeg_reference()
            
            # 5.2 run ICA and autoreject bad components (see function)
            (
                ica_data, 
                icas, 
                labels,
                excluded
            ) = run_ica_cleaning(epochs_ar,random_state=100,n_components=20)
            
            # update results dictionary
            subject_results.loc[subject, 'n_components_removed'] = len(excluded)

            # 5. save data
            ica_data.save(os.path.join(outpath, file[:-5] + '_epo.fif'), overwrite=True)
            
        # At the end of your inner loop for processing each file, append results for the file to the main results_table
            results_table = pd.concat([results_table, pd.DataFrame(subject_results)], ignore_index=True)


# After processing all subjects and their files, save the entire DataFrame to Excel
output_folder = os.path.join(project_dir,r'derivatives/scores_tms')
excel_filename = 'statistics_refresh_some.xlsx'
excel_filepath = f'{output_folder}/{excel_filename}'
results_table.to_excel(excel_filepath, sheet_name='Sheet1', index=False)
print(f"DataFrame saved to {excel_filepath}")

