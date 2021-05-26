#Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of the UFPA - ITU Artificial Intelligence/Machine Learning in 5G Challenge (http://ai5gchallenge.ufpa.br/).
#Author       		: Ailton Oliveira, Aldebaro Klautau, Wesin, Ribeiro
#Email          	: ml5gphy@gmail.com                                          
#License		: This script is distributed under "Public Domain" license.
###################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


###############################################################################
# Support functions
###############################################################################

#For description about top-k, including the explanation on how they treat ties (which can be misleading
#if your classifier is outputting a lot of ties (e.g. all 0's will lead to high top-k)
#https://www.tensorflow.org/api_docs/python/tf/nn/in_top_k
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure the files before training the net.')    
    parser.add_argument('model_name', help='Name of the model', type=str)
    #TODO: limit the number of input to 4
    parser.add_argument('-p','--plots', 
        help='Use this parametter if you want to see the accuracy and loss plots',
        action='store_true')
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

#num_test_examples = 2000  # for evaluating in the end, after training
num_validation_examples = 1960
num_training_examples = 9234
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

X_channel_train, y_train = training_generator.get_examples(num_training_examples)
X_channel_validation, y_validation = training_generator.get_examples(num_validation_examples)
X_train = X_channel_train.reshape((-1,numSamplesPerExample, 2 * Nr, 1))

print(X_train.shape)
print(y_train.shape)


# real / compl as twice number of rows
input_shape = (numSamplesPerExample, 2 * Nr, 1)
output_dim = (2 * Nr, Nt)

##############################################################################
# Model configuration
##############################################################################

#multimodal
model_name = args.model_name
model_filepath = f"models/model_{model_name}.h5"
num_epochs = 3
batch_size = 32
validationFraction = 0.2 #from 0 to 1
modelHand = ModelHandler()
opt = Adam()

model = modelHand.createArchitecture('channel_conv',np.prod(output_dim),input_shape, 'complete',output_dim)
model.compile(loss="mse",
            optimizer=opt
            #metrics=[metrics.categorical_accuracy,
                    #metrics.top_k_categorical_accuracy,
                    #top_50_accuracy]
                    )
model.summary()
hist = model.fit(X_train,y_train, 
    validation_split=validationFraction,
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-7,
            patience=5,
            # restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_filepath,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            min_delta=1e-7,
            patience=2,
            cooldown=5,
            verbose=1,
            min_lr=1e-6,
        ),
    ],)


if args.plots:
    import matplotlib.pyplot as plt
    import matplotlib     
    matplotlib.rcParams.update({'font.size': 15})

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc)+1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, 'b--', label='loss',linewidth=2)
    plt.plot(epochs, val_loss, 'g--', label='validation loss',linewidth=2)
    plt.legend()

    plt.show()