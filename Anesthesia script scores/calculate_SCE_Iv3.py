import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import pyconscious as pc
import os

# Get all subject folders
project_dir = r'/Volumes/IMADS SSD/SSD/derivatives'
eeg_folder = 'pipeline_rest'
subject_folders = os.listdir(os.path.join(project_dir, eeg_folder))

# Define output folder
output_dir = os.path.join(project_dir, 'scores_rest')

# Define standard parameters
file_extension = '_epo.fif'
condition = 'rest'

results_list = []  # This list will store dictionaries for each subject

for subject in subject_folders:
    results_table_SCE = {'subject': subject}
    
    # Get all subject data files
    data_files = os.listdir(os.path.join(project_dir, eeg_folder, subject))
    
    for file in data_files:
        if file.endswith(file_extension) and condition in file:
            
            try:
                # ... [loading and processing data] ...
                file_path = os.path.join(project_dir, eeg_folder, subject, file)
                data = mne.read_epochs(file_path)
                data.load_data()
                data.apply_baseline((None, None))
                data = mne.preprocessing.compute_current_source_density(data)
                finData = data.get_data(picks='csd')

                # 1. Calculate LZc
                resultSCE = pc.SCE(finData)
                extracted_str = file.split('_')[1:-1]
                condition_name = '_'.join(extracted_str)
                
                # Update the dictionary for this condition
                results_table_SCE[condition_name] = resultSCE

            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # After all files for this subject are processed, append the results
    results_list.append(results_table_SCE)

results_df = pd.DataFrame(results_list)

# Save the DataFrame to an Excel file
excel_filename_SCE = 'SCE_scores.xlsx'
excel_filepath_SCE = f'{output_dir}/{excel_filename_SCE}'

# Save the dataframe to Excel and insert the saved image
with pd.ExcelWriter(excel_filepath_SCE, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Sheet1', index=False)

print(f"Table saved to {excel_filepath_SCE}")
