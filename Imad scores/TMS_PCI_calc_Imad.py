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
from PCIst import pci_st
import mne_icalabel as il
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
import io
import openpyxl


# ... [other imports if any]

## Start script

# Get all subject folders
project_dir = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives'
eeg_folder = 'pipeline_TMS'
subject_folders = os.listdir(os.path.join(project_dir, eeg_folder))

# define output folder
output_dir = os.path.join(project_dir, '/pipeline_TMS')

# Define standard parameters
file_extension = '_epo.fif'
condition = 'tms'

# Initiate LZc results table
results_table_PCIst = pd.DataFrame(
    columns=[
        'subject', 'task-awake_acq-tms', 'task-sed_acq-tms_run-1', 'task-sed_acq-tms_run-2', 'task-sed_acq-tms_run-3', 'task-sed_acq-tms_run-4'
    ]
)

image_buffers = []  # To save the image buffer for each subject

for subject in subject_folders:
    
    # Get all subject data files
    data_files = os.listdir(os.path.join(project_dir, eeg_folder, subject))
    
    # loop through relevant files 
    for file in data_files:
        
        if file.endswith(file_extension) and condition in file:
            
            # create output folder for subject
            outpath = os.path.join(project_dir + output_dir, subject)
            try: 
                os.mkdir(outpath)
                print('Path created')
            except:
                print("Path exists")
            
            results_table_PCIst.loc[subject, 'subject'] = subject
            
            extracted_str = file.split('_')[1:-1]
            condition_name = '_'.join(extracted_str)
            print(condition_name)            
            
            # 1. load data (vhdr file)
            file_path = os.path.join(project_dir, eeg_folder, subject, file)
            data = mne.read_epochs(file_path)
            data.load_data()
            
            data.apply_baseline((-0.250, -0.02))
         
            evoked_data = data.average().get_data()
            par = {'baseline_window': (-0.250, -0.02), 'response_window': (0.020, 0.500), 'k': 1.2, 'min_snr': 1.1, 'max_var': 99, 'embed': False, 'n_steps': 100}
            pci = pci_st.calc_PCIst(evoked_data, data.times, **par)
            
            results_table_PCIst.loc[subject, condition_name] = pci
            
            # find peaks to plot
            # Find the index that corresponds to x=0
            start_time = 0.01  # this value might need adjustment based on your exact requirement
            start_index = np.where(data.times >= start_time)[0][0]
            
            # Take the absolute average data from the start_index onward for peak detection
            avg_abs_data = np.mean(np.abs(evoked_data[:, start_index:]), 0)
            
            # Find peaks in the modified data
            peaks, _ = mne.preprocessing.peak_finder(avg_abs_data, thresh=None, extrema=1, verbose=None)
            
            # Account for the starting index offset when plotting
            final_butterfly_plot = data.average().plot_joint(times=[data.times[p + start_index] for p in peaks], title=file)

            # Saving the image for the final butterfly plot
            fig = final_butterfly_plot.figure
            fig.axes[0].set_ylim(-10, 10)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)  # Move this line here
            #plt.close(fig)
            
            image_buffers.append((buf, file))
            
            # Save the DataFrame to an Excel file
            output_folder = '/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/tryy'
            excel_filename = 'TMS_results_tryyy.xlsx'
            excel_filepath = f'{output_folder}/{excel_filename}'

with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
     results_table_PCIst.to_excel(writer, sheet_name='Sheet1', index=False)

     # Load the workbook and access the sheet
     workbook = writer.book
     worksheet = writer.sheets['Sheet1']

     for idx, (buf, file) in enumerate(image_buffers):  # Looping through image buffers
         subject = file.split('_')[0]  # extract subject from filename
    
         # Ensure buffer isn't closed
         if not buf.closed:
             buf.seek(0)
    
             # Calculate cell_name dynamically based on DataFrame and image index
             col_letter = chr(65 + results_table_PCIst.shape[1])  # Move the image to the column after the last one
            
             # Determine the row based on the image index
             row_number = (idx * 25) + 2
            
             cell_name = f"{col_letter}{row_number}"
    
             # Add the saved image to the worksheet at the calculated position
             worksheet.add_image(openpyxl.drawing.image.Image(buf), cell_name)
    
             # Add the title above the image
             worksheet[f"{col_letter}{row_number - 1}"] = file  # setting the file name as the title
            
             print(f"Adding image for {subject} at {col_letter}{row_number}")
             print(f"Extracted subject: {subject}")

            
            
            #buf.close()  # Explicitly close the buffer

             print(f"DataFrame and Butterfly plots saved to {excel_filepath}")