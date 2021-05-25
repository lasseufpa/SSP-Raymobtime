'''
Script to extract channels from raytrace data
Authors:
Wesin Ribeiro
Marcus Yuichi
2019
'''
from scipy.io import loadmat, savemat
import numpy as np
import mimo_channels

##################################
### Script configuration
##################################

def processChannelRandomGeo(data_folder, dataset):  
    Nr = 8
    Nt = 64
    outputFile = data_folder + '/channel_data/random_geo_nominal.mat'

    #################################
    #### Start processing
    #################################

          
    import os
    import sys
    
    numRays = 2
    numEpisodes = 6500    
    Ht = np.zeros((numEpisodes, Nr,Nt))
    
    numChannels = 0
    numValidChannels = 0
    print('Processing ...')
    print(numEpisodes)
    for i in range(numEpisodes):
        numChannels += 1
        # Check valid rays for user
        numValidChannels += 1
        gain = np.random.randn(numRays) + i*np.random.randn(numRays)
        gain_in_dB = 20*np.log10(np.abs(gain))
        nominal_values = [-42.1414, 46.4871, 56.1069, 24.0976];
        angle_spread = 3;
        AoD_az = nominal_values[np.random.randint(0,3)] + angle_spread*np.random.randn(numRays);
        AoA_az = nominal_values[np.random.randint(0,3)] + angle_spread*np.random.randn(numRays);
        # AoD_az = 360*np.random.uniform(size=numRays);
        # AoA_az = 360*np.random.uniform(size=numRays);
        # RxAngle = episodeRays[iScene, i, 0:numUserRays, 8][0]
        # RxAngle = RxAngle + 90.0
        # if RxAngle > 360.0:
        #     RxAngle = RxAngle - 360.0
        # #Correct ULA with Rx orientation
        # AoA_az = - RxAngle + AoA_az #angle_new = - delta_axis + angle_wi;
        phase = np.angle(gain)*180/np.pi;
            
        Ht[i,:,:] = mimo_channels.getNarrowBandULAMIMOChannel(\
            AoD_az, AoA_az, gain_in_dB, Nt, Nr, pathPhases=phase)
            
    
        
        print('### Finished processing channels')
        print('\t %d Total of channels'%numChannels)
        print('\t %d Total of valid channels'%numValidChannels)

    
    # permute dimensions before reshape: scenes before episodes
    # found out np.moveaxis as alternative to permute in matlab
    Harray = np.reshape(Ht, (numEpisodes, Nr, Nt))
    Hvirtual = np.zeros(Harray.shape, dtype='complex128')
    scaling_factor = 1 / np.sqrt(Nr * Nt)

    for i in range(numEpisodes):
        m = np.squeeze(Harray[i,:,:])
        Hvirtual[i,:,:] = scaling_factor * np.fft.fft2(m)

    savemat(outputFile, {'Harray': Harray, 'Hvirtual':Hvirtual})
