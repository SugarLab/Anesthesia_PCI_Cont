from ITPC_functions_Imad import (
    wavelet_convolution, 
    extract_power_normalization, 
    extract_phase_calculate_ITPC,
    bootstrap_stat_dB,
    bootstrap_stat_itpc,
    threshold_dB_itpc,
    itpc_drop,
    compute_pwr_and_phase
    )
import mne
import matplotlib.pyplot as plt
import numpy as np

###Import data
awake_file = r'/Volumes/IMADS SSD/Anesthesia_conciousness_paper/derivatives/pipeline_TMS/sub-1016/sub-1016_task-awake_acq-tms_epo.fif'
a
data_awake = mne.read_epochs(awake_file)


###Subsample to only have a few channels for simplicity
data_1 = data_awake.get_data().transpose(2,1,0)[:,10:15,:] # ['P6', 'P4', 'P2', 'Pz', 'P1']


###define parameters

# for convolution 
fs_awake = = int(data_awake.info['sfreq'])

highest_frequency = 40
lowest_frequency = 5
num_wavelets = highest_frequency - lowest_frequency + 1

# for power normalization
b1 = -0.25 # baseline start
b2 = -0.05 # baseline stop

# for bootstrap statistics
n_straps = 500
alpha = 0.05

data_1_pwr_phase = compute_pwr_and_phase(data_1, data_awake, fs=fs_awake, highest_frequency=highest_frequency, lowest_frequency=lowest_frequency, num_wavelets=num_wavelets, b1=b1, b2=b2, n_straps=n_straps, alpha=alpha)

# Plot results
plt.figure()

# Plot the averaged data
plt.subplot(311)
mean_data_length = data_1.mean(axis=2).shape[0]
plt.plot(data_1.mean(axis=2))
plt.xticks(np.linspace(0, mean_data_length, 4), np.linspace(-250, 500, 4))
plt.title('awake')

# Display the first image
plt.subplot(312)
img1_length = data_1_pwr_phase[0][:,:,3].shape[1]
plt.imshow(np.squeeze(data_1_pwr_phase[0][:,:,3]))
plt.xticks(np.linspace(0, img1_length, 4), np.linspace(-250, 500, 4))

# Display the second image
plt.subplot(313)
img2_length = data_1_pwr_phase[1][:,:,3].shape[1]
plt.imshow(np.squeeze(data_1_pwr_phase[1][:,:,3]))
plt.xticks(np.linspace(0, img2_length, 4), np.linspace(-250, 500, 4))

##Sed
data_2_pwr_phase = compute_pwr_and_phase(data_2, data_ane, fs=fs_ane, highest_frequency=highest_frequency, lowest_frequency=lowest_frequency, num_wavelets=num_wavelets, b1=b1, b2=b2, n_straps=n_straps, alpha=alpha)

# Plot results
plt.figure()

# Plot the averaged data
plt.subplot(311)
mean_data_length = data_2.mean(axis=2).shape[0]
plt.plot(data_1.mean(axis=2))
plt.xticks(np.linspace(0, mean_data_length, 4), np.linspace(-250, 500, 4))
plt.title('awake')

# Display the first image
plt.subplot(312)
img1_length = data_1_pwr_phase[0][:,:,3].shape[1]
plt.imshow(np.squeeze(data_1_pwr_phase[0][:,:,3]))
plt.xticks(np.linspace(0, img1_length, 4), np.linspace(-250, 500, 4))

# Display the second image
plt.subplot(313)
img2_length = data_1_pwr_phase[1][:,:,3].shape[1]
plt.imshow(np.squeeze(data_1_pwr_phase[1][:,:,3]))
plt.xticks(np.linspace(0, img2_length, 4), np.linspace(-250, 500, 4))
