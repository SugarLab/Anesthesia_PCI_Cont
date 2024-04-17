import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Define paths and parameters
project_dir = Path("/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives")
eeg_folder = 'pipeline_itpc'
output_dir = project_dir / 'statistics' / 'itpc_average'
file_extension = '.pkl'
condition = 'tms'

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize DataFrame
df_results = pd.DataFrame()

# Iterate over subject folders
for subject in os.listdir(project_dir / eeg_folder):
    
    subject_path = project_dir / eeg_folder / subject
    
    # Check if it's a folder
    if subject_path.is_dir():
        
        print(f"Processing: {subject_path}")
        
        # Initialize a dictionary to store the data for the subject
        subject_data = {"Awake": np.nan, "Sed_1": np.nan, "Sed_2": np.nan, "Sed_3": np.nan, "Sed_4": np.nan}
        
        # loop through files in the subject folder 
        for file_name in os.listdir(subject_path):
            
            if file_name.endswith(file_extension) and condition in file_name:
                
                print(f"Found file: {file_name}")
                
                # Load data
                with open(subject_path / file_name, 'rb') as f:
                    itpc_data = pickle.load(f)
                
                # Calculate the average itpc_drop
                avg_itpc_drop = np.mean(itpc_data['first_n_itpc_drop'], axis=0)                
                
                # Check the condition and store the result
                if "awake" in file_name:
                    subject_data["Awake"] = avg_itpc_drop
                elif "sed" in file_name:
                    # Extract the sedation number from the filename
                    sed_num = [s.split('-')[-1] for s in file_name.split('_') if 'run' in s][0]
                    subject_data[f"Sed_{sed_num}"] = avg_itpc_drop
        
        # Append the results of the current subject to the DataFrame
        df_results = pd.concat([df_results, pd.DataFrame(subject_data, index=[subject])])

# Save the results DataFrame to an Excel file
df_results.to_excel(output_dir / 'average_first_n_itpc_drop_alpha_values.xlsx', index=True, engine='openpyxl')
