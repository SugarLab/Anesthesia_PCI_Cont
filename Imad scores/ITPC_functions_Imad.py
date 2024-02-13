# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:07:46 2023

@author: bjorneju (adapted from notebook 5 in https://github.com/chrihoni/University-of-Oslo_PCI_Mouse_Analysis.git)
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal

def wavelet_convolution (data1, lowest_frequency = 1, highest_frequency = 1000, num = 50, fs = 1000., X1 =-0.25, X2=0.5):
    """returns convolution array (frequ x samples x trials x channels) from input data (samples x channels x trials) 
       and Morlet wavelet family (frequency x samples). 
    
        %% WAVELET CONVOLUTION 
        # A wavelet convolution is perfomed on each  trial  in order to
        # extract phases for all time-frequency points.
        # 1) The kernel of the convolution is created (family of wavelets)
        # 2) Convolution is performed
        # 3) Power and Phases are extracted for each time-frequency-channel-trial points
        Parameters
        ----------
        lowest_frequency: lowest wavlet freq (int, float), default = 1

        highest_frequency: highest wavlet freq (int, float), default = 1000

        scale: scale the number of wavelet frequencies. If lowest_frequency and highest_frequency are e.g. 1 and 10 then
        10 wavelets are created by using scale =1. If scale = 2 the number of wavelets is divided by 2. --> corresponding to freq 1,3,5,7,9
    """
    
    ###1) create family of wavelets   
    start = X1 # was originally -2 and 2
    stop  = abs(X1)
    wavelet_time = np.linspace(start, stop, num = int((stop-start) * fs + 1))    #wavelet length
    n = 3   #wavelet cycles
    num_wavelets = num  #number of pk (frequency bands)
    
    # create a vector of peak frequencies:
    frequencies = np.linspace(lowest_frequency, highest_frequency,num_wavelets) 
    
    # initialize the matrix of wavelet family
    wavelet_family = np.zeros((num_wavelets, len(wavelet_time)), dtype = complex)
     
    # Loop through frequencies and make a family of wavelets
    for fi in range (num_wavelets):
        # create a sine wave at this frequency
        sinewave = np.exp(2j * np.pi * frequencies[fi] * wavelet_time)     # the "j" makes it a complex wavelet
        # create a Gaussian window
        gaus_win = np.exp(-wavelet_time**2 / (2*(n/(2*np.pi*frequencies[fi]))**2))
        #create wavelet via element-by-element multiplication of the sinewave and gaussian window
        wavelet_family[fi,:] = sinewave * gaus_win    
    
    
    ###2) CONVOLUTION: loop through frequencies and compute synchronization### --> Matrix: fr(wavelet) x data(data1) x sweeps(data1) ch(data)
    prestim = abs(X1)  #in [s], define max abs time pre-stimulus
    
    data1 = data1 [int((fs*X1)+(fs*prestim)):int((fs*X2)+(fs*prestim)+1),:,:] # define subset data for CONVOLUTION
    
    # pad signal with reversed baseline before signal and reveresed end of signal (length of baseline) after signal
    # padding length
    N_padding = int(np.abs(fs * X1))
    # Extracting and reversing the first N datapoints
    start_padding = data1[:N_padding, :, :][::-1]
    
    # Extracting and reversing the last N datapoints
    end_padding = data1[-N_padding:, :, :][::-1]
    
    # Concatenating the reversed portions to the original data
    padded_data = np.concatenate([start_padding, data1, end_padding], axis=0)
    
    ### CONVOLUTION: loop through frequencies and compute synchronization###
    conv = np.zeros((np.size(wavelet_family, axis = 0),np.size(padded_data, axis = 0),np.size(data1, axis = 2),np.size(data1,axis = 1)), dtype = complex)
    
    
    for fr in range(np.size(wavelet_family,0)):
        for sw in range(np.size(data1,2)):
            for ch in range(np.size(data1,1)):
        
                conv[fr,:,sw,ch] = sp.signal.convolve(np.squeeze(padded_data[:,ch,sw]), np.squeeze(wavelet_family[fr,:]), mode = "same") 

    
    time_data = np.linspace(X1, X2, num = int((X2-X1) * fs +1 ))    #time data
    return conv[:,N_padding:-N_padding,:,:], fs, X1, X2, frequencies, time_data, wavelet_family[:,N_padding:-N_padding]
   

    
def extract_power_normalization(conv, fs, X1, X2, b1 = -0.5, b2 = -0.2):
    """returns powers_NORM, decibels and decibels_M by extracting POWERS and performing NORMALIZATION from conv returned by ""wavelet_convolution" function
    Parameters:
    -----------
    
    b1: calculate mean baseline over b1 and b2--> b1 (start) in sec, e.g. -0.5s (default)
    
    b2: calculate mean baseline over b1 and b2--> b2 (stop) in sec, e.g. -0.2s (default) 
    """
    ###global powers_NORM, decibels, decibels_M
    ###EXTRACTS POWERS and PERFORMS NORMALIZATION:###
    
    # Raw power extraction
    powers = (abs(conv[:,:,:,:])**2) # the array contains: frequency x samples x trials x channels
    
    # Power Normalization: from uV^2 to % of baseline to dB:
    
    # Mean baseline across time for each trial, channel and frequency band:
    power_baseline = np.squeeze(np.mean(powers[:,int((fs*b1)+(fs*abs(X1))): int((fs*b2)+(fs*abs(X1))+1),:,:], axis = 1)) #identical values in Matlab vs Numpy array: x:y(Matlab) == (x-1):y (Numpy) 
    # Mean baseline across time and trials for each frequency band and channel:
    power_baseline = np.squeeze(np.mean(power_baseline,axis = 1))
    
    
    # Normalization on the mean baseline across time and trials  
    powers_NORM = np.zeros((np.size(powers,axis = 0),np.size(powers,axis = 1),np.size(powers,axis = 2),np.size(powers,axis = 3)))
    
    for fr in range(np.size(powers,axis = 0)): # frequencies
        for sw in range(np.size(powers,axis = 2)): # trials
            for ch in range(np.size(powers,axis =3)): # channels
        
      # Normalization of each trial on the mean baseline across trials and time for each freq. band and channel
                powers_NORM [fr,:,sw,ch] =  powers[fr,:,sw,ch] / power_baseline[fr,ch]
    
    
    del fr, sw, ch 
    del power_baseline
    
    
    # Mean Normalized Power across trials
    powers_M_NORM = np.squeeze(np.mean (powers_NORM, axis = 2))
    
    # Power Conversion to dB
    decibels =  10 * (np.log10(powers_NORM))
    decibels_M =  10 * (np.log10(powers_M_NORM))
    
    return powers_NORM, decibels, decibels_M


def extract_phase_calculate_ITPC(conv):
    """returns phases and av_vector_length by extracting PHASE and computing inter-trial-phase-clustering (ITPC) from conv returned by "wavelet_convolution" function
    
    """
    ###global phases, av_vector_length
    ### PHASE EXTRACTION and CLUSTERING COMPUTATION:
    
    # Raw Phase Extraction
    phases = np.angle(conv[:,:,:,:])    # exctract phases from convolution
    
    # Compute the average vector of phases of each trials at all time points and peak frequencies
    # by Euler's formula: M *np.exp(1j*phase angle); (the Magnitude M is not taken into consideration, hence all vector lenths are set to  1)
    # NB: av_vector_length= magnitude of phase-locking across trials [between 0 and 1]
    
    av_vector_length = np.squeeze(abs(np.mean(np.exp(1j * (phases[:,:,:,:])), axis = 2))) # Inter Trial Phase Clustering
    
    return phases, av_vector_length


def bootstrap_stat_dB(powers_NORM, fs, X1, b1 = -0.5, b2 = -0.2, n_straps = 500, alpha = 0.05):
    """returns bootstrap_dB_max and bootstrap_dB_min, the max and min baseline powers of the bootstraped distribution across trials. powers_norm are returned by "extract_power_normalization" function
    
    Parameters
    ----------
    b1: default = -0.5 -->  time range of baseline in sec, same as in "extract_power_normalization"
    
    b2: default = -0.2 -->  time range of baseline in sec, same as in "extract_power_normalization"
    
    n_straps: default = 500 --> total number of resamplings (bootstraps; usually between 500 and 1000 is enough)
    
    alpha: default = 0.05   --> significant level (probability of rejecting the null hypothesis when it's true) can be 0,05 or less
    
    """    
    # BOOTSTRAP STATISTIC
    #Bootstrap statistic is performed in order to conserve only the significant variations from baseline
    #of both spectral power and phase-locking in response to the stimulation:
    #a) It finds 1 positive and 1 negative thresholds based on the surrogate distribution of the maximum and
    #minimum of the averaged normalized spectral power (across resampled trials)
    #for each frequency and channel (threshold at alpha = 0.05)
    #b) It finds 1 positive threshold based on the surrogate distribution
    #of the maximum Inter Trial Phase Clustering (across resampled trials)
    #for each frequency and channel (threshold at alpha = 0.01)
    #c) It performs thresholding on both powers (dB signal) and phase-locking (ITPC signal)
    #in order to conserve only significant variations from baseline
    
    
    #a) COMPUTE STATISTICAL THRESHOLD FOR dB
    
    # DEFINE PARAMETERS:
    #global T_max_percentile_dB, T_min_percentile_dB
    db1 = b1    #[s] time range of baseline
    db2 = b2
    baseline = powers_NORM[:,   int((fs*db1)+(fs*abs(X1))):int((fs*db2)+(fs*abs(X1))+1),  :,  :]  #baseline matrix from which it calculates the distribution (Matrix: frequency x samples x trials x channels)
    n_samples = np.size(baseline, axis = 1) # number of samples
    n_sweeps = np.size(baseline, axis = 2)  # number of trials
    n_freq = np.size(baseline,axis = 0)    # number of frequencies
    n_channel = np.size(baseline,axis = 3) # number of channels
    
    
    # LOOP OVER BOOTSTRAP SAMPLES AND CREATE SURROGATE DISTRIBUTIONS 
    #Background: Bootstrap is generally useful for estimating the distribution of a statistic (e.g. mean, variance) 
    #without using normal theory (e.g. z-statistic, t-statistic). 
    #Bootstrap comes in handy when there is no analytical form or normal theory to help estimate 
    #the distribution of the statistics of interest, since bootstrap methods can apply to most random quantities,
    #e.g., the ratio of variance and mean. (https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
    
    bootstrap_dB_max = np.zeros((n_freq,n_channel,n_straps))
    bootstrap_dB_min = np.zeros((n_freq,n_channel,n_straps))
    
    for ii in range (n_straps):
        
            # Take randomly (n = n_sweeps) samples from data (indexes of sweeps, n_sweeps) 
            # with replacement:
            resampled_sweeps = np.random.choice(np.arange(0,n_sweeps),n_sweeps) 
            
            # Take the actual trials corresponding to the randomized indexes (Matrix: frequency x samples x trials x channels):
            resampled_baseline_dB = baseline[:,:,resampled_sweeps,:]
             
            # calculate mean over trials of the new resampled baseline matrix and dB conversion (frequency x samples x trials x channels):
            resampled_averageBaseline_dB = np.squeeze(np.mean(resampled_baseline_dB,axis =2))
            resampled_averageBaseline_dB = 10 * (np.log10(resampled_averageBaseline_dB))
       
            bootstrap_dB_max[:,:,ii] = np.max(resampled_averageBaseline_dB,axis = 1) #max from samples (axis = 1)
            bootstrap_dB_min[:,:,ii] = np.min(resampled_averageBaseline_dB, axis = 1) #min from samples (axis = 1)


    # FIND THRESHOLDS
    #Background: A percentile (or a centile) is a measure used in statistics indicating the value below 
    #which a given percentage of observations in a group of observations falls. 
    #For example, the 20th percentile is the value (or score) below which 20% of the observations may be found. 
    #Equivalently, 80% of the observations are found above the 20th percentile. 
    #https://en.wikipedia.org/wiki/Percentile#:~:text=A%20percentile%20(or%20a%20centile,the%20observations%20may%20be%20found.
    
    T_max_percentile_dB = np.zeros((n_freq,n_channel)) # Positive threshold
    T_min_percentile_dB = np.zeros((n_freq,n_channel)) # Negative threshold
    
    for ch in range (n_channel):
        for fr in range (n_freq):
            T_max_percentile_dB[fr,ch] = np.percentile(bootstrap_dB_max[fr,ch,:],100*(1-alpha))
            T_min_percentile_dB[fr,ch] = np.percentile(bootstrap_dB_min[fr,ch,:],100*alpha)
        
    
    del baseline, n_straps, alpha, n_samples, n_sweeps, n_freq, n_channel
    del resampled_averageBaseline_dB,  resampled_baseline_dB, resampled_sweeps
    del bootstrap_dB_max, bootstrap_dB_min,  db1, db2
    
    return T_max_percentile_dB, T_min_percentile_dB


def bootstrap_stat_itpc (phases, fs, X1, b1 = -0.5, b2 =-0.2, n_straps = 500, alpha = 0.05):
    """returns bootstrap_ITPC_max, the max baseline ITPC values of the bootstraped distribution across trials. input argument phases is returned by "extract_phase_calculate_ITPC" function
    Parameters
    ---------
    b1: default = -0.5 -->  time range of baseline in sec, same as in "extract_power_normalization"
    
    b2: default = -0.2 -->  time range of baseline in sec, same as in "extract_power_normalization"
    
    n_straps: default = 500 --> total number of resamplings (bootstraps; usually between 500 and 1000 is enough)
    
    alpha: default = 0.05   --> significant level (probability of rejecting the null hypothesis when it's true) can be 0,05 or less

    """
    
    ###b) COMPUTE STATISTICAL THRESHOLD FOR ITPC
      
    
    # DEFINE PARAMETERS:
    #global bootstrap_ITPC_max, T_max_percentile_ITPC
        
    p1 = b1    # [s] time range of baseline
    p2 = b2
    
    # Take the phases of baseline
    Phases_baseline = phases[:,int((fs*p1)+(fs*abs(X1))):int((fs*p2)+(fs*abs(X1))+1),:,:]
    
    # Euler's Formula:
    Euler_baseline = (np.exp(1j*(Phases_baseline[:,:,:,:]))) # Partial Inter Trial Phase Clustering formula (Matrix: frequency x samples x trials x channels)
    
    # Define Boostrap Parameters:
    baseline = Euler_baseline #% baseline matrix from which it calculates the surrogate distribution (Matrix: frequency x samples x trials x channels)
    n_sweeps = np.size (baseline,axis =2)  # number of sweeps
    n_freq = np.size(baseline,axis = 0)     # number of frequencies
    n_chii = np.size(baseline,axis = 3)  # number of channels
    
    
    # LOOP OVER BOOTSTRAP SAMPLES AND CREATE SURROGATE DISTRIBUTIONS 
    
    bootstrap_ITPC_max = np.zeros((n_freq,n_chii,n_straps))
    
    for bb in range (n_straps):
        
            # Take randomly (n = n_sweeps) samples from data (indexes of sweeps, 1:nsweeps), 
            # with replacement:
            resampled_sweeps = np.random.choice(np.arange(0,n_sweeps),n_sweeps) 
            
            # Take the actual sweeps corresponding to the randomized indexes:
            resampled_baseline = baseline[:,:,resampled_sweeps,:]
            
            # Make the Inter Trial Phase Clustering (ITPC) of the new resempled baseline matrix:
            resampled_ITPC_baseline = np.squeeze(abs(np.mean(resampled_baseline,axis = 2)))
       
           
            # Find the maximum value of the new ITPC of
            # running baseline and create a matrix with MAX for each
            # frequecy and channel:
            bootstrap_ITPC_max[:,:,bb] = np.max(resampled_ITPC_baseline, axis = 1) 
           
    del bb
    
    
    
    # FIND THRESHOLDS
    
    T_max_percentile_ITPC = np.zeros((n_freq, n_chii))
    for chii in range (n_chii):
        for fr in range (n_freq):
            T_max_percentile_ITPC[fr,chii] = np.percentile(bootstrap_ITPC_max[fr,chii,:],100*(1-alpha))
            
    del chii,fr
    del  n_sweeps, n_freq
    del resampled_sweeps
    del bootstrap_ITPC_max
    del Euler_baseline, n_chii, Phases_baseline, resampled_ITPC_baseline, resampled_baseline
    
    return T_max_percentile_ITPC

def threshold_dB_itpc (decibels_M, T_max_percentile_ITPC, T_min_percentile_dB, T_max_percentile_dB, av_vector_length):
    """returns the significant normalized mean powers(dB) and clustered unit-length vectors (ITPC --> see also analyzing EEG time series p. 243/244). Input dB- and itpc- data  are returned by 'bootstrap_stat_dB' and 'bootstrap_stat_itpc' functions"""
    #### THRESHOLDING ON dB ####
    # TAKE THE SIGNIFICANT MEAN RELATIVE POWERS (dB)
    
    # Set not significant values to zero for decibels_M.
    #decibels_M = is mean normalized power spectrum, for freq(fr) x samples(samp) x channels (ch)
    #global decibels_M_sig, av_vector_length_sig
    
    decibels_M_sig = np.zeros((np.size(decibels_M, axis = 0), np.size(decibels_M, axis = 1), np.size(decibels_M,axis = 2)))
    
    for fr in range (np.size(decibels_M, axis = 0)):
        for samp in range (np.size(decibels_M, axis = 1)):
            for ch in range (np.size(decibels_M, axis = 2)):
       
                if decibels_M[fr, samp, ch] > T_max_percentile_dB[fr,ch]:  
                    decibels_M_sig[fr, samp, ch]  =      decibels_M[fr, samp, ch] 
                
                elif decibels_M[fr, samp, ch]  <  T_min_percentile_dB[fr,ch]: 
                    decibels_M_sig[fr, samp, ch]  =  decibels_M[fr, samp, ch] 
                
                else: 
                    decibels_M_sig[fr, samp, ch] = 0
            
          
    del fr, samp, ch
    
    #### THRESHOLDING ON ITPC####
    # TAKE THE SIGNIFICANT ITPC from av_vector_length -->  freq(fr) x samples(samp) x channels (ch)
    
    av_vector_length_sig = np.zeros((np.size(av_vector_length,axis = 0), np.size(av_vector_length, axis = 1), np.size(av_vector_length, axis = 2)))
    
    for fr in range (np.size(av_vector_length, axis = 0)):
        for samp in range (np.size(av_vector_length, axis = 1)):
            for ch in range (np.size(av_vector_length, axis = 2)):
       
                if av_vector_length[fr, samp, ch] > T_max_percentile_ITPC[fr,ch]:   
                    av_vector_length_sig[fr, samp, ch] = av_vector_length[fr, samp, ch]
    
                else:
                    av_vector_length_sig[fr, samp, ch] = 0 
        
    
    del fr, samp, ch
    return decibels_M_sig, av_vector_length_sig



def itpc_drop(data1, time_data, fs, frequencies, ITPC_post_x1 = 0, ITPC_post_x2 = 0.8, fH_start = 8, fH_end =40):
    """
    returns npy arrays:
        
    ITPC_drop :   timepoint of last significant ITPC value in the specified time and frequency range for each channel
    
    ITPC_drop_M : mean timepoint of last significant ITPC value in the specified time and frequency range across all channels
    
    ITPC_freq:    the specified frequency range    
    
    PARAMETERS 
    ----------
    data: input data   --> "av_vector_length_sig" returned by threshold_dB_itpc (freq x samples x channels) 
    
    
    ITPC_post_x1 = 0 [s]   --> the range of the interval, start
    
    ITPC_post_x2 = 0.8 [s] --> the range of the interval, stop
    
    
    fH_start = 8  [Hz]     --> lower limit of High Frequency range for calculating mean ITPC (frequency must be present in frequencies array)
    
    fH_end = 40   [Hz]     --> upper limit of High Frequency range for calculating mean ITPC (frequency must be present in frequencies array)
    
        
    
    """
    #% Post stimulus time window to use:
    
    
     
    
    #start value of time_data (returned by wavelet_convlutin function): prestim   
    prestim = np.abs(time_data[0]) #[s]
    
    
    #start sample of ITCP_post_x1: start_sample
    start_sample = int(fs * ITPC_post_x1) + int(fs * prestim)
    
    #stop sample of ITCP_post_x2: stop_sample
    stop_sample  = int(fs * ITPC_post_x2) + int(fs * prestim)
    
    
    #find indices of fH_start fH_end in frequencies array: fH_x1, fH_x2 
    fH_x1 = np.where(frequencies == fH_start)[0][0]
    fH_x2 = np.where(frequencies == fH_end)[0][0]
    
    
    #% Extract the poststimulus matrix of the significant ITCP (average vector % lengths) for all frequencies and channels: 
    ITPC_postim_H_sig = data1 [fH_x1:fH_x2, start_sample:stop_sample+1, :]
    #% Average across frequencies, axis = 0: ITPC_postim_av_H (array: samples, channels) 
    ITPC_postim_av_H = np.mean(ITPC_postim_H_sig,axis =0)
    
    #% Create its time vector:
    time_postim = np.linspace(ITPC_post_x1, ITPC_post_x2, num = int((ITPC_post_x2 - ITPC_post_x1) * fs+1))
    
    
     
    #%
    #% ITPC_drop Time FOR EACH CHANNEL
    #%
    
    
    #% Find the sample point corresponding to the last synchronous time point
    #% whithin the defined frequency range for each channel:
        
     
    #create empty array for appending significant ITPC values: ITPC_index_lastsig   
    ITPC_index_lastsig = np.zeros(np.size(ITPC_postim_av_H, axis =1))
    
    for ch in range (np.size(ITPC_postim_av_H, axis =1)):   
     
    #% IF no phase clustering between 0 and .8s (end) -> No synchrony at all -> set to first sample (0s)
        if np.mean(ITPC_postim_av_H[:,ch], axis = 0) == 0:
            ITPC_index_lastsig[ch] = np.where(time_postim==ITPC_post_x1)[0][0]
        else:
        #% FIND the last significant ITPC in the range    
            ITPC_index_lastsig[ch] = np.where(ITPC_postim_av_H[:,ch])[0][-1]
    
    
       
    #% the coressponding timepoint of index ITPC_index_lastsig in time_postim for each channel: ITPC_drop
    ITPC_drop = time_postim[ITPC_index_lastsig.astype(int)]
    
    #% the mean coressponding timepoint of index ITPC_index_lastsig in time_postim across all channels: ITPC_drop_M
    ITPC_drop_M = np.mean(ITPC_drop)
    
    ITPC_freq = [fH_start, fH_end]
    
    return ITPC_drop, ITPC_drop_M, ITPC_freq
    




def early_freq_power(data1, time_data, fs, frequencies, x1=0.008, x2=0.018, fH_start =20, fH_end=40):

    """
        returns npy arrays:
            
            OFF_dB :       mean power of early high freq. response for each channel (array: channels)
            
            OFF_dB_M :     mean power of early high freq. response across all channels (array: scalar)
            
            OFF_frange:    the specified frequency range (array: frequencies)
            
            dB_HF_allCH:   mean of decibels_M_sig across across frequency range (array: samples, channels)
        
        PARAMETERS 
        ----------
        data: input data   --> "decibels_M_sig" returned by threshold_dB_itpc (freq x samples x channels) 
        
        
        x1 = 0.008 [s]   --> the range of the interval after stim, start
        
        x2 = 0.018 [s] -->  the range of the interval after stim, stop
        
        
        fH_start = 8  [Hz]     --> lower limit of High Frequency range for calculating mean power (frequency must be present in frequencies array)
        
        fH_end = 40   [Hz]     --> upper limit of High Frequency range for calculating mean power (frequency must be present in frequencies array)
        
               
        """
    
    ## QUANTIFICATION OF EARLY HIGH FREQUENCY POWER (OFF PERIOD, if any)
    #% It quantifies the High Frequency power dynamic early after the stimulation
    #% and detects the OFF period (suppression of high frequencies) if present
    
    #%
    #% MEAN HF POWER AFTER STIMULATION (OFF period, if HF power < 0dB)
    #% NB: Obtained from each channel and then averaged
    #%
    
    #% Post stimulus time window to use:
    
    #start value of time_data (returned by wavelet_convlutin function): prestim   
    prestim = np.abs(time_data[0]) #[s]  
    
    #start sample of x1: start_sample
    start_sample = int(fs * x1) + int(fs * prestim)+1
    
    #stop sample of  x2: stop_sample
    stop_sample  = int(fs * x2) + int(fs * prestim)+1
    
    
        
    #find indices of fH_start, fH_end: fH_x1, fH_x2   
    
    
    fH_x1 = np.where(frequencies == fH_start)[0][0]
    fH_x2 = np.where(frequencies == fH_end)[0][0]
    
    
    #mean of decibels_M_sig across across frequency range, fH_start, fH_end : dB_HF_allCH (array: samples, channels)
    #note: db conversion back into % of baseline (10**(....)/10)
    dB_HF_allCH = np.mean(10**(data1[fH_x1:fH_x2, :, :] / 10), axis =0)
    
    #mean of dB_HF_allCH across samples: OFF_dB_p  (array: channels)
    OFF_dB_p = np.mean(dB_HF_allCH [start_sample : stop_sample, :], axis =0)
    
    #mean of OFF_dB_p across channels: OFF_dB_p  (scalar)
    OFF_dB_m_p = np.mean(OFF_dB_p)
    
    
    #convert OFF_dB_p back into dB: OFF_dB 
    OFF_dB = 10*np.log10(OFF_dB_p)
    
    #convert OFF_dB_m_p back into dB: OFF_dB_M 
    OFF_dB_M = 10*np.log10(OFF_dB_m_p)
    
    
    OFF_frange = np.array([fH_start, fH_end])
    
    return OFF_dB, OFF_dB_M, OFF_frange, dB_HF_allCH
    



def late_freq_power(decibels_M_sig, time_data, fs, frequencies, x1=0.1, x2=0.7, fH_start =20, fH_end=80):

    """
        returns:
            
            peak_time :     time of late hig freq. response peak averaged across channels (scalar) 
            
            last_sig_time : time of last siginificant high freq. power averaged across channels (scalar)    
            
                    
        PARAMETERS 
        ----------
        data: input data   --> "decibels_M_sig" returned by threshold_dB_itpc (freq x samples x channels) 
        
        
        x1 = e.g.  0.1 [s]   --> the range of the interval after stim, start
        
        x2 = e.g.  0.7 [s] -->  the range of the interval after stim, stop
        
        
        fH_start = e.g. 20  [Hz]     --> lower limit of High Frequency range for calculating mean power (frequency must be present in frequencies array)
        
        fH_end =   e.g. 80  [Hz]     --> upper limit of High Frequency range for calculating mean power (frequency must be present in frequencies array)
                       
        """
    
    
    #parameters for function testing 
    data1 = decibels_M_sig
    x1 = 0.1
    x2 = 0.7
    fH_start = 20
    fH_end = 80
    
    #% Post stimulus time window to use:
    
    #start value of time_data (returned by wavelet_convlutin function): prestim   
    prestim = np.abs(time_data[0]) #[s]  
    
    #start sample of x1: start_sample
    start_sample = int(fs * x1) + int(fs * prestim)+1
    
    #stop sample of  x2: stop_sample
    stop_sample  = int(fs * x2) + int(fs * prestim)+1
    
            
    #find indices of fH_start, fH_end: fH_x1, fH_x2     
    fH_x1 = np.where(frequencies == fH_start)[0][0]
    fH_x2 = np.where(frequencies == fH_end)[0][0]
    
    #mean of decibels_M_sig across across frequency range, fH_start, fH_end : dB_HF_allCH1 (array: samples, channels)
    #note: db conversion back into % of baseline (10**(....)/10)
    dB_HF_allCH1 = np.mean(10**(data1[fH_x1:fH_x2, :, :] / 10), axis =0)

    
    #find late high freq power peak in data1 (samples x channels) for each channel: peak
    peak = np.argmax(dB_HF_allCH1[start_sample : stop_sample, :], axis =0)
    #slice time_data ranging from start_sample to stop_sample: time_sl
    time_sl = time_data[start_sample : stop_sample]
    #find timepoint of peak in time_sl: time_peak
    time_peak = time_sl[peak]
    #take mean of time_peak: time_peak_mean
    time_peak_mean = np.mean(time_peak)
    
    #find last significant power increase for each channel:ix_bool
    ix_bool = dB_HF_allCH1[start_sample : stop_sample, :] >1
    #loop over channel axis (axis 1) and append index of last significant value (i.e. where ix == True): ix_last
    ix_last = [np.where(ix_bool[:,i] == True)[0][-1] for i in range(np.size(dB_HF_allCH1, axis=1))]
    #find corresponding time for ix_last: time_last
    time_last = time_sl[ix_last]
    #mean across channels of time_last: time_last_mean
    time_last_mean = np.mean(time_last)
    
    return time_peak_mean, time_last_mean
    

def min_power(data1, time_data, fs, x1_H = 0.0, x2_H = 0.3):
    """returns:
            
   MIN_power, MIN_time : min power, time of high freq. response for each channel (array: channels)
            
   MIN_power_mean, MIN_time_mean : mean min power, time of high freq. response across all channels (scalar)
         
           
        
    PARAMETERS 
    ----------
    data: input data   --> "dB_HF_allCH" returned by early_freq_power function;
    -->mean of decibels_M_sig across across frequency range (array: samples, channels)
    
    
    x1_H = 0.0 [s] -->  the range of the interval after stim, start
    
    x2_H = 0.3 [s] -->  the range of the interval after stim, stop   
    """
    
    #%
    #% QUANTIFICATION OF:
    #% SUPPRESSION MIN (in High Frequency Range) - Minimum Peak
    #% SUPPRESSION end and start (in High Frequency Range)  - Starting and ending point of OFF period
    #%
    
    #% NB: Obtained from each channel and then averaged
    
    
    
    #% Post stimulus time window to use:
        
    #start value of time_data (returned by wavelet_convlutin function): prestim   
    prestim = np.abs(time_data[0]) #[s]
    
    
    #start sample of x1_H: start_sample
    start_sample = int(fs * x1_H) + int(fs * prestim)
    
    #stop sample of  x2_H: stop_sample
    stop_sample  = int(fs * x2_H) + int(fs * prestim)+1
    
    
    #% HIG FREQ range;
    
    #% Extract matrix
    T = time_data[start_sample : stop_sample]
    
    
    ##mean HF power after stimulation, in time range x1_H, x2_H : P (array: samples x channels)
    #% reconverted in dB: P
    P = 10* (np.log10(data1[start_sample: stop_sample, :]))
    
    
    
    #% Percentage 
    #Perc = dB_HF_allCH (((fs.* x1_H)+(fs.*prestim)+1):((fs.* x2_H)+(fs.*prestim)+1),:);
    
    ##return the minimum power for index and channel in the specified HF range and time range: MIN_index, channel
    MIN_index, channel = np.where(P == np.min(P, axis =0))
    
                                  
    ##create empty numpy.array for appending the min power index for each channel (the 1st if several): MIN_index1 (channels)
    MIN_index1 = np.array([]).astype(int)
    ##create empty numpy.array for appending the min power for each channel (the 1st if several): MIN_power (channels)
    MIN_power  = np.array([])
    ##appending the min power index for each channel (the 1st if several): MIN_index1
    for i in range (np.size(P, axis = 1)):    
        ##store index for minimum power value for each channel (1 st value if several): arr
        arr = np.array([MIN_index[ix] for ix, value in enumerate(channel) if value == i])[0]
        ##append arr: MIN_index1 
        MIN_index1 = np.append(MIN_index1, arr)
        ##append min power for each channel: MIN_power
        MIN_power  = np.append(MIN_power, P[arr, i]) 
    
    ##find corresponding time of index: MIN_time (channels)
    MIN_time =  T[MIN_index1]
    
    #% Obtain the mean minimum power and time among channels:
    MIN_power_mean = 10*(np.log10(np.mean(10**((MIN_power)/10))))
    MIN_time_mean = np.mean(MIN_time)
   
    return MIN_power, MIN_time, MIN_power_mean, MIN_time_mean








def late_power(data1, time_data, fs, ITPC_d, t1=0.1, t2=0.8, fdb1=20, fdb2 =40):


    #% LATE HF POWER QUANTIFICATION
    #
    #% FIND THE TIME POINT OF FIRST HFpower AFTER SUPPRESSION for each channel
    #% FIND ALSO THE TIME POINT OF THE CORRESPONDING ITPC
    #% MAKE AVERAGE ACROSS THOSE CHANNELS
    #% CALCULATE PERSENTAGE OF CHANNELS THAT HAVE HFpower AFTER SUPPRESSION
    #%
    
    """
    returns:
            
     Latepower_time_mean:  mean time of late HFpower onset (scalar)
     
     Latepower_time:       time of late HFpower onset (array: channels)
     
     Latepower_ITPC_mean:  mean ITPC drop with late HFpower (scalar)
     
     Latepower_ITPC:       ITPC drop  with late HFpower (array:channels)
         
           
        
    PARAMETERS 
    ----------
    data: input data   --> "decibels_M_sig" returned by threshold_dB_itpc (freq x samples x channels)
        
    
    t1 = 0.1 [s] -->  the range of the interval after stim, start
    
    t2 = 0.8 [s] -->  the range of the interval after stim, stop  
    
            
    fdb1 = 20    --> frequency window [Hz], start
    fdb2 = 40    --> stop
    
    """
    global power_wind_time, power_wind_logic
    
    
    #start value of time_data (returned by wavelet_convlutin function): prestim   
    prestim = np.abs(time_data[0]) #[s]
    
    
    #start sample of x1_H: start_sample
    start_sample = int(fs * t1) + int(fs * prestim)
    
    #stop sample of  x2_H: stop_sample
    stop_sample  = int(fs * t2) + int(fs * prestim)+1
    
    
    
    #% Extract matrix
    T = time_data[start_sample : stop_sample]
    
    power_wind = data1[:,start_sample : stop_sample,:]
    
    
    #% NB: conversion into % of baseline
    power_wind_NORM = np.mean(10**((power_wind[fdb1:fdb2,:,:]/10)),axis = 0)
    
    #% NB: reconverted in dB
    power_wind_dB = 10*(np.log10(power_wind_NORM))
    
    
    power_wind_logic = power_wind_dB > 0
    
    power_wind_index = np.zeros(np.size(power_wind_logic, axis =1))
    
    
    for ii in range(np.size(power_wind_logic, axis =1)):
        if np.mean(power_wind_logic[:, ii], axis = 0) == 0:
            power_wind_index[ii] = 0       
        
        
        else:
            power_wind_index[ii] = np.where(power_wind_logic[:,ii] ==1)[0][0]
        
    
    
    
    power_channels_aaa = power_wind_index > 0
    
    power_channels = np.where(power_channels_aaa)[0]#; % channels with significant HFpower
    
    power_wind_time_aaa = power_wind_index[power_channels].astype(int)
    
    power_wind_time = T[power_wind_time_aaa]#; % onset time of significant HFpower
    
    power_channel_ITPC = ITPC_d[power_channels]#; % ITPCdrop of channels with significant HFpower
    
    #% Results:
    Latepower_time_mean = np.mean(power_wind_time)#;% mean time of late HFpower onset
    Latepower_time = power_wind_time
    
    Latepower_ITPC_mean = np.mean(power_channel_ITPC)#; % mean ITPC drop of channel with late HFpower 
    Latepower_ITPC = power_channel_ITPC
    
    #if len(Latepower_time) >=1:
        #Latepower_time = Latepower_time[0]
    
    #Latepower_chpersent = size(power_channels,2)/size(data,2); % persentage of channels with  late HF power
    return Latepower_time_mean, Latepower_time,  Latepower_ITPC_mean, Latepower_ITPC





def plot_dB_itpc (data, frequencies, decibels_M_sig, av_vector_length_sig, time_data, cortex_area, x1 = .2, x2 =  .6, y1 = -400, y2 =  300, s1 = -8, s2 = +8, ch =  0, ch_prb=0, sw1 = 7, f1 = 1, f2 = 40, z1 = -5, z2 = +5, itpc1 = 0, itpc2 = .5, ITPC_freq = [8, 40],HF_frange = [20, 40]):
    """It plots one figure:
            
    _ [UPPER LEFT, RIGHT] The mean ERP from 1 channel with 5 single trials overimposed, plus the same channel in bold, in the butterfly plot with the mean ERPs from all other channels.
    
    _ [MIDDLE LEFT, RIGHT] The mean power spectrum [dB] from one channel and the average across a specified frequency range for all the channels
    
    _[LOWER LINE LEFT, RIGHT] The phase-locking across trials [ITPC] from one channel and the average across specified frequency range for all the channels
    
    Parameters
    ----------
     # Single channel PARAMETERS:
    x1 = -.2  # [s]   Time to plot (start)
    
    x2 =  .6  # [s]   Time to plot (end)
    
    y1 = -400 # [uV]  Amplitude range (start)
    
    y2 =  300 # [uV]  Amplitude range (end)
    
    s1 = -8   # [dB]  Power range (start)
    
    s2 = +8   # [dB]  Power range (end)
    
    ch =  0   # number of channel to visualize, as specified in loaded data array 
    
    ch_prb =  # ch order as indicated in prbfile_ch_order
    
    sw1 = 7   # number of trial from which it visualizes 5 consecutive trials
    
    f1 = 1    # frequency to plot (start)
    
    f2 = 40   # frequency to plot (end)

    
    # All channels PARAMETERS:
           
    z1 = -5     # [dB]  Power range (start)
    
    z2 = +5     # [dB]  Power range (start)
    
    itpc1 = 0   # ITPC  Phase-locking range (start)
    
    itpc2 = .5  # ITPC  Phase-locking range (end)
    
    
    ### Operation PARAMETERS:
        
    ITPC_freq = [8, 40]     # frequency range to use to calculate mean Phase-locking, in [Hz]
    
    HF_frange = [20, 40]    # frequency range to use to calculate mean High Frequency power, in [Hz]
    
    
    
    """
        
    # OPERATIONS:
    
    #HF_frange = [20, 40]  
    
    MEDIA1 = np.squeeze(np.mean(data, axis = 2))
    
    # Define start and end of the range:
    f_start_ITCP = ITPC_freq[0]
    f_end_ITCP = ITPC_freq[1]
        
    f_start_dB = HF_frange[0]
    f_end_dB = HF_frange[1]
        
    # bottom frequency limit for average ITPC (start) [Hz]
    fi_logic_start_ITCP = frequencies==f_start_ITCP
    fi_index_start_ITCP = np.flatnonzero(fi_logic_start_ITCP)[0]
    
    # upper frequency limit for average ITPC (end)  [Hz]
    fi_logic_end_ITCP = frequencies==f_end_ITCP
    fi_index_end_ITCP = np.flatnonzero(fi_logic_end_ITCP)[0]
    
    # bottom frequency limit for average dB (start) [Hz]
    fi_logic_start_dB = frequencies==f_start_dB
    fi_index_start_dB = np.flatnonzero(fi_logic_start_dB)[0]
    
    # upper frequency limit for average dB (end)  [Hz]
    fi_logic_end_dB = frequencies==f_end_dB
    fi_index_end_dB = np.flatnonzero(fi_logic_end_dB)[0]
    
    
    # Obtain the average dB across frequencies:
    av_dB_w = 10*(np.log10(np.mean(10**((decibels_M_sig [fi_index_start_dB : fi_index_end_dB,:,:])/10), axis = 0))) #dB to power conversion in inner parentheses: for averaging over power not dB--> convert then again to dB values  
    
    # Obtain the average ITPC across frequencies:
    av_vector_length_M_0w = np.squeeze(np.mean(av_vector_length_sig[fi_index_start_ITCP:fi_index_end_ITCP,:,:],axis = 0))
    
    
    ###PLOT:
    
    # SINGLE CHANNEL:
    
    # PLOT 5 ERP SINGLE SWEEPS and Mean. (Upper left)
    # Wakefulness:
    #clear previous figure
        
    
    fig, ax = plt.subplots(nrows =3, ncols =2, figsize=((17,12)))
    plt.subplots_adjust(wspace =0.5, hspace = 0.5)
    
    ###set size of ticks
    ax[0,0].tick_params(axis='both', labelsize=25)
        
    
    ax[0,0].plot (time_data, data[:, ch, sw1], 'tab:gray', linewidth=0.5 )
    ax[0,0].plot(time_data, data [:, ch, sw1+1], 'tab:gray', linewidth=0.5 )
    ax[0,0].plot(time_data, data [:, ch, sw1+2], 'tab:gray', linewidth=0.5)
    ax[0,0].plot (time_data, data [:, ch, sw1+3], 'tab:gray', linewidth=0.5)                                  
    ax[0,0].plot (time_data, data [:, ch, sw1+4],'tab:gray', linewidth=0.5)
    mean_trials = np.mean(data[:,:,:], axis = 2)
    ax[0,0].plot(time_data, mean_trials[:,ch], 'k')
    
    ax[0,0].set_xlabel('Time [s]', fontsize=25) 
    ax[0,0].set_ylabel('Amplitude [uV]', fontsize=25)
    ax[0,0].set_xlim(x1,x2)   
    ax[0,0].set_ylim(y1,y2)   
    ax[0,0].set_title('Mean ERP and 5 single ERPs from Channel ' + str(ch)) #ch number according to prb.file -->e.g. ch 0 = is ch12 (chnr 1) in group 0 
    #img = plt.imshow(np.random.random((100, 100)))
    #plt.colorbar(img, ax=ax[0,0])
    
          
    # PLOT SIGNIFICANT spectral power (dB) AT EACH FREQUENCY  (Middle left)
        ###set size of ticks
    ax[1,0].tick_params(axis='both', labelsize=25)

    
    cf = ax[1,0].contourf(time_data,frequencies,decibels_M_sig[:,:,ch],100, vmin = s1, vmax= s2)
    cbaxes = fig.add_axes([0.45, 0.4, 0.02, 0.19]) 
    m = plt.cm.ScalarMappable()
    m.set_array(decibels_M_sig[:,:,ch])
    m.set_clim(s1, s2)
    cb = plt.colorbar(m, cax = cbaxes, boundaries = np.linspace(s1, s2, 41)) 
    cb.ax.tick_params(labelsize=15)
    ax[1,0].set_xlabel('Time [s]', fontsize=25) 
    ax[1,0].set_ylabel('Frequencies [Hz]', fontsize=25)
    ax[1,0].set_xlim(x1,x2)
    ax[1,0].set_ylim(f1,f2)
    ax[1,0].set_yticks([2,16,32,48,64,80])
    ax[1,0].set_title('Spectral power [dB] from Channel ' + str(ch))
    
    
    
    ax[2,0].tick_params(axis='both', labelsize=25)

    # PLOT ITPC OF ALL FREQUENCY BANDS IN TIME (Lower left)
    cf1 = ax[2,0].contourf(time_data,frequencies,av_vector_length_sig[:,:,ch],100, vmin=itpc1, vmax=itpc2)
    #fig.add_axes([left/right, up/down, width, height]) 
    cbaxes = fig.add_axes([0.45, 0.11, 0.02, 0.19]) 
    m2 = plt.cm.ScalarMappable()
    m2.set_array(av_vector_length_sig[:,:,ch])
    m2.set_clim(itpc1, itpc2)
    cb1 = plt.colorbar(m2, cax = cbaxes,boundaries = np.linspace(itpc1, itpc2, 41), format='%.1f')  
    cb1.ax.tick_params(labelsize=15)
    ax[2,0].set_xlabel('Time [s]', fontsize=25)
    ax[2,0].set_ylabel('Frequencies [Hz]', fontsize=25)
    ax[2,0].set_xlim(x1,x2)   
    ax[2,0].set_ylim(f1,f2) 
    ax[2,0].set_yticks([2,16,32,48,64,80])
    ax[2,0].set_title('Inter Trial Phase Clusering [ITPC] from channel ' + str(ch))   
       
    
    
    ### ALL CHANNELS 
    ax[0,1].tick_params(axis='both', labelsize=25)

    # Plot Mean ERPs for all channels, and 1 channel highlighted in bold (Upper right)
    ax[0,1].plot (time_data, MEDIA1[:, :], 'tab:gray', linewidth=0.5 )
    ax[0,1].plot(time_data, MEDIA1[:,ch], 'k')
    ax[0,1].set_xlabel('Time [s]', fontsize=25) 
    ax[0,1].set_ylabel('Amplitude [uV]', fontsize=25)
    ax[0,1].set_xlim(x1,x2)   
    ax[0,1].set_ylim(y1,y2)   
    ax[0,1].set_title('Mean ERPs for all Channels (ch' + str(ch) + ' [' + cortex_area[0][ch]+']' +  (' in bold)'))
    
    
    
    # PLOT dB FOR EACH CHANNEL IN TIME (Middle right)
    ax[1,1].tick_params(axis='x', labelsize=25)
    ax[1,1].tick_params(axis='y', labelsize=15)
    av_dB_w_resh = np.transpose(av_dB_w)
    channels = np.arange(1, np.size(av_dB_w, axis =1)+1, dtype = float)  
    cf2 = ax[1,1].contourf(time_data,channels, av_dB_w_resh[:,:],100, vmin = z1, vmax = z2)
    m1 = plt.cm.ScalarMappable()
    m1.set_array(av_dB_w_resh[:,:])
    m1.set_clim(z1, z2)
    cbaxes = fig.add_axes([0.92, 0.4, 0.02, 0.19]) 
    cb2 = plt.colorbar(m1, cax = cbaxes, boundaries = np.linspace(z1, z2, 41), format='%.1f')
    cb2.ax.tick_params(labelsize=15)
    ax[1,1].set_xlabel('Time [s]', fontsize= 25) 
    ax[1,1].set_ylabel('Ctx area', fontsize=25)
    ax[1,1].set_yticks(np.arange(1,np.size(av_dB_w, axis =1)+1))
    ax[1,1].set_yticklabels(cortex_area[0])

    ax[1,1].set_xlim(x1,x2)
    #ax[1,1].set_ylim(1, np.size(av_dB_w, axis =1))
    ax[1,1].set_title('Mean Power [dB] between ' + str(round(f_start_dB, ndigits=2)) + ' & ' + str(round(f_end_dB, ndigits=2)) + ' Hz for all channels')
    
    
    # PLOT ITPC FOR EACH CHANNEL IN TIME
    ax[2,1].tick_params(axis='x', labelsize=25)
    ax[2,1].tick_params(axis='y', labelsize=15)

    av_vector_length_M_0w_resh = np.transpose(av_vector_length_M_0w)
    channels = np.arange(1, np.size(av_dB_w, axis =1)+1, dtype = float)  
    cf3 = ax[2,1].contourf(time_data,channels, av_vector_length_M_0w_resh[:,:],100, vmin =itpc1, vmax=itpc2)
    cbaxes = fig.add_axes([0.92, 0.11, 0.02, 0.19]) 
    m3 = plt.cm.ScalarMappable()
    m3.set_array(av_vector_length_M_0w_resh[:,:])
    m3.set_clim(itpc1, itpc2)
    cb3 = plt.colorbar(m3, cax = cbaxes, boundaries= np.linspace(itpc1, itpc2, 41), format='%.1f' )
    cb3.ax.tick_params(labelsize=15)
    ax[2,1].set_xlabel('Time [s]', fontsize=25) 
    ax[2,1].set_ylabel('Ctx area', fontsize=25)
    ax[2,1].set_yticks(np.arange(1,np.size(av_dB_w, axis =1)+1))
    ax[2,1].set_yticklabels(cortex_area[0])

    ax[2,1].set_xlim(x1,x2)
    #ax[2,1].set_ylim(1, np.size(av_dB_w, axis = 1))
    ax[2,1].set_title('Mean ITPC between ' + str(round(f_start_ITCP, ndigits=2)) + ' & ' + str(round(f_end_ITCP, ndigits=2)) + ' Hz for all channels')



def mm_to_inch(value):
    """converts mm to inch (for use in e.g. plt.figure(figsize=(mm_to_inch(20), mm_to_inch(10))"""
    return (value/25.4)

def compute_pwr_and_phase(data_1, time_data, fs, highest_frequency, lowest_frequency, num_wavelets, b1, b2, n_straps, alpha):
    # Awake data
    ###convolute data with wavelets: conv 
    conv, fs, X1, X2, frequencies, time_data, wavelet_family = wavelet_convolution(data_1, lowest_frequency = lowest_frequency, highest_frequency = highest_frequency, num = num_wavelets, fs = fs)

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
    return decibels_M_sig, av_vector_length_sig
    