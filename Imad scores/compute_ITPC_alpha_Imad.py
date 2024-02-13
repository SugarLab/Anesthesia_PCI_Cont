# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:23:23 2023

@author: bjorneju
"""

from ITPC_functions_Imad_v_2_0 import itpc_for_paper
import mne
import os
import pickle

## Start script

# Get all subject folders
project_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives'
eeg_folder = 'pipeline_TMS'
subject_folders = os.listdir(os.path.join(project_dir, eeg_folder))

# define output folder
output_dir = os.path.join(project_dir, 'pipeline_itpc_alpha_band')

# Define standard parameters
file_extension = '_epo.fif'
condition = 'tms'

for subject in subject_folders:
    
    # Get all subject data files
    data_files = os.listdir(os.path.join(project_dir, eeg_folder, subject))
    
    # loop through relevant files 
    for file in data_files:
        
        if file.endswith(file_extension) and condition in file:
            
            # create output folder for subject
            outpath = os.path.join(output_dir, subject)
            try: 
                os.mkdir(outpath)
                print('Path created')
            except:
                print("Path exists")
                
            # 1. load data (vhdr file)
            file_path = os.path.join(project_dir, eeg_folder, subject, file)
            epoch_data = mne.read_epochs(file_path)

            # Band-pass filter
            epoch_data.filter(l_freq=8, h_freq=13, method='iir', picks=['eeg','eog'])
            
            data = epoch_data.get_data().transpose(2,1,0)
            fs = epoch_data.info['sfreq']
            
            # compute itpc values
            itpc_d, between_n_itpc_d, last_n_itpc_d, first_n_itpc_d, gmfp, last_n_gmfp, first_n_gmfp = itpc_for_paper(
                data, fs, lowest_frequency = 5, highest_frequency=40, 
                b1 = -0.25, b2 = -0.05, n_straps = 500, alpha = 0.05, 
                itpc_post_x1=0, itpc_post_x2=0.5, fH_start=8, fH_end=40, n=15
            )
            
            itpc_data = {
                'itpc_drop': itpc_d, 
                'last_n_itpc_drop': last_n_itpc_d, 
                'first_n_itpc_drop': first_n_itpc_d, 
                'gmfp': gmfp, 
                'last_n_gmfp': last_n_gmfp, 
                'first_n_gmfp': first_n_gmfp
            }
            
            # save data
            with open(os.path.join(outpath,file[:-4]+'.pkl'), 'wb') as f:
                pickle.dump(itpc_data, f)
                
