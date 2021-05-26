#Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of the UFPA - ITU Artificial Intelligence/Machine Learning in 5G Challenge (http://ai5gchallenge.ufpa.br/).
#Author       		: Wesin Alves, Ailton Oliveira, Aldebaro Klautau, Arthur Nascimento, Diego Gomes, Jamelly Ferreira, Walter Frazao
#Email          	: ml5gphy@gmail.com                                          
#License		: This script is distributed under "Public Domain" license.
###################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Trains a deep NN for choosing top-K beams
Adapted by AK: Aug 7, 2018
See
https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
and
https://stackoverflow.com/questions/45642077/do-i-need-to-use-one-hot-encoding-if-my-output-variable-is-binary
See for explanation about convnet and filters:
https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d
and
http://cs231n.github.io/convolutional-networks/
'''
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model
from tensorflow.keras.layers import Dense,concatenate, Reshape
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta,Adam
from sklearn.model_selection import train_test_split
from ModelHandler import ModelHandler
from mimo_channels_data_generator2 import RandomChannelMimoDataGenerator
import numpy as np
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure the files before training the net.')
    parser.add_argument('model_name', help='Name of the model', type=str)    
    args = parser.parse_args()

###############################################################################
# Data configuration
###############################################################################
tf.device('/device:GPU:0')
tgtRec = 3


###############################################################################
# Channel data configuration
# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

# training parameters
epochs = 100

# Parameters
global Nt
global Nr
Nt = 64  # num of Rx antennas, will be larger than Nt for uplink massive MIMO
Nr = 8  # num of Tx antennas
# the sample is a measurement of Y values, and their collection composes an example. The channel estimation
min_randomized_snr_db = -1
max_randomized_snr_db = 1

# must be done per example, each one having a matrix of Nr x numSamplesPerExample of complex numbers
numSamplesPerExample = 256  # number of channel uses, input and output pairs
# if wants a gradient calculated many times with same channel
numExamplesWithFixedChannel = 1
numSamplesPerFixedChannel = (
    numExamplesWithFixedChannel * numSamplesPerExample
)  # coherence time
# obs: it may make sense to have the batch size equals the coherence time
batch = 1  # numExamplesWithFixedChannel

num_test_examples = 1960
channel_train_input_file = f"../SSP_data/ce_baseline_data/{args.model_name}.mat"
print("Reading dataset... ",channel_train_input_file)
method = "manual"

# Generator
training_generator = RandomChannelMimoDataGenerator(
    batch_size=batch,
    Nr=Nr,
    Nt=Nt,
    # num_clusters=num_clusters,
    numSamplesPerFixedChannel=numSamplesPerFixedChannel,
    # numSamplesPerExample=numSamplesPerExample, SNRdB=SNRdB,
    numSamplesPerExample=numSamplesPerExample,
    # method='random')
    method=method,
    file = channel_train_input_file
)
if True:
    training_generator.randomize_SNR = True
    training_generator.min_randomized_snr_db = min_randomized_snr_db
    training_generator.max_randomized_snr_db = max_randomized_snr_db
else:
    training_generator.randomize_SNR = True
    training_generator.SNRdB = 0



##############################################################################
# Model configuration
##############################################################################

#multimodal
model_name = args.model_name
model_filepath = f"models/model_{model_name}.h5"
logdir = 'results/'
model = keras.models.load_model(model_filepath)

SNRdB_values = np.arange(-21, 22, 3)
training_generator.randomize_SNR = False
training_generator.method = "manual"
print(model.summary())


all_nmse_db_average = np.zeros((SNRdB_values.shape))
all_nmse_db_min = np.zeros((SNRdB_values.shape))
all_nmse_db_max = np.zeros((SNRdB_values.shape))

it = 0
for SNRdB in SNRdB_values:
    training_generator.SNRdB = SNRdB
    # get rid of the last example in the training_generator's memory (flush it)
    X_channel_test, outputs = training_generator.get_examples(1)
    # now get the actual examples:
    X_channel_test, outputs = training_generator.get_examples(num_test_examples)
    X_channel_test = X_channel_test.reshape((-1,numSamplesPerExample, 2 * Nr, 1))    

    predictedOutput = model.predict(X_channel_test)
    error = outputs - predictedOutput
    mseTest = np.mean(error[:] ** 2)
    print("overall MSE = ", mseTest)
    mean_nmse = mseTest / (Nr * Nt)
    print("overall NMSE = ", mean_nmse)
    nmses = np.zeros((num_test_examples,))
    for i in range(num_test_examples):
        this_H = outputs[i]
        this_error = error[i]
        nmses[i] = np.mean(this_error[:] ** 2) / np.mean(this_H[:] ** 2)

    print("NMSE: mean", np.mean(nmses), "min", np.min(nmses), "max", np.max(nmses))
    nmses_db = 10 * np.log10(nmses)
    print(
        "NMSE dB: mean",
        np.mean(nmses_db),
        "min",
        np.min(nmses_db),
        "max",
        np.max(nmses_db),
    )

    all_nmse_db_average[it] = np.mean(nmses_db)
    all_nmse_db_min[it] = np.min(nmses_db)
    all_nmse_db_max[it] = np.max(nmses_db)

    it += 1

    del X_channel_test
    del outputs
    #del mat 

output_filename = (
    f"all_nmse_{model_name}.txt"
)
output_filename = os.path.join(logdir, output_filename)
np.savetxt(output_filename, (all_nmse_db_average, all_nmse_db_min, all_nmse_db_max))
print("Wrote file", output_filename)
print("*******************\n{}".format(np.mean(all_nmse_db_average)))