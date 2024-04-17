from ITPC_functions import (
    wavelet_convolution, 
    extract_power_normalization, 
    extract_phase_calculate_ITPC,
    bootstrap_stat_dB,
    bootstrap_stat_itpc,
    threshold_dB_itpc,
    itpc_drop
    )
import mne
import matplotlib.pyplot as plt
import numpy as np

###Import data
awake_file = r'\\hypatia.uio.no\lh-med-imb-jstormlab\Data\Anesthesia_Project\EEG_Analysis\EEG_Preprocessed\sub-1016_task-awake_acq-tms_epo.fif'
ane_file = r'\\hypatia.uio.no\lh-med-imb-jstormlab\Data\Anesthesia_Project\EEG_Analysis\EEG_Preprocessed\sub-1016_task-sed_acq-tms_run-1_epo.fif'
data_awake = mne.read_epochs(awake_file)
data_ane = mne.read_epochs(ane_file)

###Subsample to only have a few channels for simplicity
data_1 = data_awake.get_data().transpose(2,1,0)[:,10:15,:] # ['P6', 'P4', 'P2', 'Pz', 'P1']
data_2 = data_ane.get_data().transpose(2,1,0)[:,10:15,:] # ['P6', 'P4', 'P2', 'Pz', 'P1']

###define parameters

# for convolution 
fs = int(data_awake.info['sfreq'])
highest_frequency = 40
lowest_frequency = 5
num_wavelets = highest_frequency - lowest_frequency + 1

# for power normalization
b1 = -0.25 # baseline start
b2 = -0.05 # baseline stop

# for bootstrap statistics
n_straps = 500
alpha = 0.05

# Plot results
plt.figure()
plt.subplot(311)
plt.plot(data_1.mean(axis=2))
plt.xlim([0,750])
plt.title('awake')
plt.subplot(312)
plt.imshow(np.squeeze(decibels_M_sig[:,:,3]))
plt.subplot(313)
plt.imshow(np.squeeze(av_vector_length_sig[:,:,3]))



## Sed data
###convolute data with wavelets: conv 
conv, fs, X1, X2, frequencies, time_data, wavelet_family = wavelet_convolution(data_2, lowest_frequency = lowest_frequency, highest_frequency = highest_frequency, num = num_wavelets, fs = fs)

###extracting POWERS and performing NORMALIZATION from conv: powers_NORM, decibels and decibels_M 
powers_NORM, decibels, decibels_M = extract_power_normalization(conv, fs, X1, X2, b1 = b1, b2 = b2)

###extracting PHASE and computing inter-trial-phase-clustering (ITPC) from conv returned by "wavelet_convolution" function: phases and av_vector_length
phases, av_vector_length = extract_phase_calculate_ITPC(conv)

#returns the max and min baseline powers of the bootstraped distribution across trials. powers_norm are returned by "extract_power_normalization" function: bootstrap_dB_max and bootstrap_dB_min
T_max_percentile_dB, T_min_percentile_dB = bootstrap_stat_dB(powers_NORM, fs, X1, b1 = b1, b2 = b2, n_straps = n_straps, alpha = alpha)

#returns the max ITPC value of the bootstraped distribution across trials. Input argument phases is returned by "extract_phase_calculate_ITPC" function: bootstrap_ITPC_max, the max baseline ITPC values
T_max_percentile_ITPC = bootstrap_stat_itpc (phases, fs, X1, b1 = b1, b2 = b2, n_straps = n_straps, alpha = alpha)

#returns the significant normalized mean powers(dB) and clustered unit-length vectors: decibels_M_sig, av_vector_length_sig 
#Input dB- and itpc- data  are returned by 'bootstrap_stat_dB' and 'bootstrap_stat_itpc' functions
decibels_M_sig, av_vector_length_sig = threshold_dB_itpc (decibels_M, T_max_percentile_ITPC, T_min_percentile_dB, T_max_percentile_dB, av_vector_length)

#itpc drop in 8 to 40 hz range (broadband)
ITPC_d, ITPC_d_m, ITPC_fr = itpc_drop(av_vector_length_sig, time_data, fs, frequencies, ITPC_post_x1 = 0, ITPC_post_x2 = 0.5, fH_start = 8, fH_end = 40)

# Plot results
plt.figure()
plt.subplot(311)
plt.plot(data_2.mean(axis=2))
plt.xlim([0,750])
plt.title('Anesthesia')
plt.subplot(312)
plt.imshow(np.squeeze(decibels_M_sig[:,:,3]))
plt.subplot(313)
plt.imshow(np.squeeze(av_vector_length_sig[:,:,3]))