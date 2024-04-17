#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:04:53 2024

@author: imadjb
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:23:23 2023

@author: bjorneju
"""

import mne
import os
import pickle
import numpy as np

## Start script

# Get all subject folders
project_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives'
eeg_folder = 'pipeline_TMS'
subject_folders = os.listdir(os.path.join(project_dir, eeg_folder))

# define output folder
output_dir = os.path.join(project_dir, 'pipeline_gmfp')

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
            
            data = epoch_data.get_data().transpose(2,1,0)
            
            # compute gmfp values
            gmfp = np.mean(np.abs(np.mean(data,axis=2)),axis=1)
            last_n_gmfp = np.mean(np.abs(np.mean(data[:,:,-15:],axis=2)),axis=1)
            first_n_gmfp = np.mean(np.abs(np.mean(data[:,:,:15],axis=2)),axis=1)
            
            gmfp_data = {
                 'gmfp': gmfp, 
                 'last_n_gmfp': last_n_gmfp, 
                 'first_n_gmfp': first_n_gmfp
            }
            
            # save data
            with open(os.path.join(outpath,file[:-4]+'.pkl'), 'wb') as f:
                pickle.dump(gmfp_data, f)
               
