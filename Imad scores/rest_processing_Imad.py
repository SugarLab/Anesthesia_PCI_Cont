import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import pyconscious as pc
import os
from autoreject import AutoReject
import mne_icalabel as il

## Define functions used in script
# ICA cleaning
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

# Epoch data
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


## Start script

# Get all subject folders
project_dir = r'/Volumes/IMADS SSD/SSD/'
eeg_folder = 'EEG'
subject_folders = os.listdir(os.path.join(project_dir,eeg_folder))

# define output folder
output_dir = os.path.join(project_dir,'derivatives/pipeline_rest')

# Define standard parameters
file_extension = 'vhdr'
condition = 'rest'

# Initiate results dictionary
results_dictionary = []

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
                  
            results_table = {
            'subject': subject,
            'raw_data_file': file,
            }
            
            # update results dictionary
            results_table['subject'] = subject
        
            # 1. load data (vhdr file)
            file_path = os.path.join(project_dir,eeg_folder,subject,file)
            data = mne.io.read_raw_brainvision(file_path)
            data.load_data()
            
            # 1.1. channel info (remove EMG and set type for EOG channels)
            if data.info['ch_names'][-1] == 'EMG':
                data.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog', 'EMG': 'emg'})
                # Drop EMG channel if present (might have forgotten to change workspace during recording)
                data.drop_channels(data.info['ch_names'][-1])
            else:
                data.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})
            data.set_montage('standard_1005')
            
            # 1.2 Crop data to only contain first 5 minutes
            if data.times.max()>300.:
                data.crop(tmin=0, tmax=300.)
            
            # 2. resample (with low-pass filter!) and filter data
            new_sampling = 1000.
            hp_freq = 0.5
            lp_freq = 40
            
            # resample
            data.resample(new_sampling, npad='auto')
            
            # Band-pass filter
            data.filter(l_freq=hp_freq, h_freq=lp_freq, method='fir', picks=['eeg','eog'])
            
            # 3. regress out eye artefacts
            # Add reference channel for regression to work
            eog_data = mne.add_reference_channels(data,['ref'],copy=True)
            eog_data,_ = mne.set_eeg_reference(eog_data,['ref'])
            # fit regression
            weights = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog').fit(eog_data)
            # apply regression to correct eye artefacts
            eog_data_clean = weights.apply(eog_data, copy=True)
            eog_data_clean.drop_channels(['ref'])

            # 4. clean data using autoreject
            # 4.1 Epoch data
            epochs = epoch(eog_data_clean)
            
            # 4.2 Apply autoreject
            ar = AutoReject(verbose='tqdm',picks='eeg')
            ar.fit(epochs)  # fit on a few epochs to save time
            epochs_ar, reject_log = ar.transform(epochs, return_log=True)
            
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
            results_table['pct_segments_interpolated'] = interpolated_segments/total_segments
            results_table['pct_segments_rejected'] = rejected_segments/total_segments
            results_table['n_segments'] = total_segments
            results_table['pct_trials_rejected'] = sum(reject_log.bad_epochs)/len(reject_log.bad_epochs)
            results_table['n_trials'] = len(reject_log.bad_epochs)

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
            results_table['n_components_removed'] = len(excluded)
            
            # 5. save data
            ica_data.save(os.path.join(outpath, file[:-5] + '_epo.fif'), overwrite=True)
            
            results_dictionary.append(results_table)
            
new_results = pd.DataFrame(results_dictionary)           
            
# Save the DataFrame to an Excel file
excel_filename = 'Spontanuous_statistics.xlsx'
excel_filepath = f'{output_dir}/{excel_filename}'

            
# Save the dataframe to Excel and insert the saved image
with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
     new_results.to_excel(writer, sheet_name='Sheet1', index=False)

print(f"DataFrame and Butterfly plot saved to {excel_filepath}")            
            
            
            
