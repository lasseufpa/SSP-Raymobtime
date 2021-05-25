#Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of the UFPA - ITU Artificial Intelligence/Machine Learning in 5G Challenge (http://ai5gchallenge.ufpa.br/).
#Author       		: Ailton Oliveira, Aldebaro Klautau, Arthur Nascimento, Diego Gomes, Jamelly Ferreira, Walter Frazao
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


###############################################################################
# Support functions
###############################################################################

#For description about top-k, including the explanation on how they treat ties (which can be misleading
#if your classifier is outputting a lot of ties (e.g. all 0's will lead to high top-k)
#https://www.tensorflow.org/api_docs/python/tf/nn/in_top_k
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def beamsLogScale(y,thresholdBelowMax):
        y_shape = y.shape
        
        for i in range(0,y_shape[0]):            
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs + 1e-30)
            minValue = np.amax(logOut) - thresholdBelowMax
            zeroedValueIndices = logOut < minValue
            thisOutputs[zeroedValueIndices]=0
            thisOutputs = thisOutputs / sum(thisOutputs)
            y[i,:] = thisOutputs
        
        return y

def getBeamOutput(output_file):
    
    thresholdBelowMax = 6
    
    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']
    
    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]
    
    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y = beamsLogScale(y,thresholdBelowMax)
    
    return y,num_classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure the files before training the net.')
    parser.add_argument('data_folder', help='Location of the data directory', type=str)
    parser.add_argument('model_name', help='Name of the model', type=str)
    #TODO: limit the number of input to 4
    parser.add_argument('--input', nargs='*', default=['coord'], 
        choices = ['img', 'coord', 'lidar','channel'],
        help='Which data to use as input. Select from: img, lidar or coord.')
    parser.add_argument('-p','--plots', 
        help='Use this parametter if you want to see the accuracy and loss plots',
        action='store_true')
    args = parser.parse_args()
else:
    import sys
    sys.path.append('../submission_baseline_example/')
    import common
    args = common.args

###############################################################################
# Data configuration
###############################################################################
tf.device('/device:GPU:0')
data_dir = args.data_folder+'/'
tgtRec = 3

if 'coord' in args.input: 
    ###############################################################################
    # Coordinate configuration
    #train
    coord_train_input_file = data_dir+'coord_input/coord_train.npz'
    coord_train_cache_file = np.load(coord_train_input_file)
    X_coord_train = coord_train_cache_file['coordinates']
    # #validation
    # coord_validation_input_file = data_dir+'coord_input/coord_validation.npz'
    # coord_validation_cache_file = np.load(coord_validation_input_file)
    # X_coord_validation = coord_validation_cache_file['coordinates']

    coord_train_input_shape = X_coord_train.shape

if 'img' in args.input:
    ###############################################################################
    # Image configuration
    resizeFac = 20 # Resize Factor
    nCh = 1 # The number of channels of the image
    imgDim = (360,640) # Image dimensions
    method = 1
    #train
    img_train_input_file = data_dir+'image_input/img_input_train_'+str(resizeFac)+'.npz'
    print("Reading dataset... ",img_train_input_file)
    img_train_cache_file = np.load(img_train_input_file)
    X_img_train = img_train_cache_file['inputs']
    #validation
    # img_validation_input_file = data_dir+'image_input/img_input_validation_'+str(resizeFac)+'.npz'
    # print("Reading dataset... ",img_validation_input_file)
    # img_validation_cache_file = np.load(img_validation_input_file)
    # X_img_validation = img_validation_cache_file['inputs']

    img_train_input_shape = X_img_train.shape

if 'lidar' in args.input:
    ###############################################################################
    # LIDAR configuration
    #train
    lidar_train_input_file = data_dir+'lidar_input/lidar_train.npz'
    print("Reading dataset... ",lidar_train_input_file)
    lidar_train_cache_file = np.load(lidar_train_input_file)
    X_lidar_train = lidar_train_cache_file['input']
    #validation
    # lidar_validation_input_file = data_dir+'lidar_input/lidar_validation.npz'
    # print("Reading dataset... ",lidar_validation_input_file)
    # lidar_validation_cache_file = np.load(lidar_validation_input_file)
    # X_lidar_validation = lidar_validation_cache_file['input']

    lidar_train_input_shape = X_lidar_train.shape

if 'channel' in args.input:
    ###############################################################################
    # Channel data configuration
    # fix random seed for reproducibility
    seed = 1
    np.random.seed(seed)

    # Parameters
    global Nt
    global Nr
    Nt = 16  # num of Rx antennas, will be larger than Nt for uplink massive MIMO
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
    channel_train_input_file = data_dir + f"channel_data/{args.model_name}.mat"
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
    #input_shape = (numSamplesPerExample, 2 * (Nr))
    input_shape = (numSamplesPerExample, 2 * Nr, 1)
    #input_shape = (numSamplesPerExample, Nr)
    output_dim = (2 * Nr, Nt)

    # numInputs = np.prod(input_shape)
    # numOutputs = np.prod(output_dim)
    # print(numInputs, " ", numOutputs)

###############################################################################
# Output configuration
#train
# output_train_file = data_dir+'beam_output/beams_output_train.npz'
# y_train,num_classes = getBeamOutput(output_train_file)

# output_validation_file = data_dir+'beam_output/beams_output_validation.npz'
# y_validation, _ = getBeamOutput(output_validation_file)

##############################################################################
# Model configuration
##############################################################################

#multimodal
model_name = args.model_name
model_filepath = f"models/model_{model_name}.h5"

multimodal = False if len(args.input) == 1 else len(args.input)

num_epochs = 100
batch_size = 32
validationFraction = 0.2 #from 0 to 1
modelHand = ModelHandler()
opt = Adam()

if 'coord' in args.input:
    coord_model = modelHand.createArchitecture('coord_mlp',np.prod(output_dim),coord_train_input_shape[1],'complete', output_dim)
if 'img' in args.input:
    #num_epochs = 5
    if nCh==1:   
        img_model = modelHand.createArchitecture('light_image',np.prod(output_dim),[img_train_input_shape[1],img_train_input_shape[2],1],'complete', output_dim)
    else:
        img_model = modelHand.createArchitecture('light_image',np.prod(output_dim),[img_train_input_shape[1],img_train_input_shape[2],img_train_input_shape[3]],'complete', output_dim)
if 'lidar' in args.input:
    lidar_model = modelHand.createArchitecture('lidar_marcus',np.prod(output_dim),[lidar_train_input_shape[1],lidar_train_input_shape[2],lidar_train_input_shape[3]],'complete', output_dim)
if 'channel' in args.input:
    channel_model = modelHand.createArchitecture('lidar_marcus',np.prod(output_dim),input_shape, 'complete',output_dim)

if multimodal == 2:
    if 'channel' in args.input and 'lidar' in args.input:
        combined_model = concatenate([channel_model.output, lidar_model.output])
        z = Dense(Nt,activation="linear")(combined_model)
        model = Model(inputs=[channel_model.input,lidar_model.input],outputs=z)
        model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
        model.summary()
        hist = model.fit([X_channel_train,X_lidar_train],y_train, 
            validation_split=0.2,
            epochs=num_epochs,batch_size=batch_size,
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
    
    elif 'channel' in args.input and 'img' in args.input:
        combined_model = concatenate([channel_model.output, img_model.output])
        z = Dense(Nt,activation="linear")(combined_model)
        model = Model(inputs=[channel_model.input,img_model.input],outputs=z)
        model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
        model.summary()
        hist = model.fit([X_channel_train,X_img_train],y_train, 
            validation_split=0.2,
            epochs=num_epochs,batch_size=batch_size,
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
    elif 'channel' in args.input and 'coord' in args.input:
        combined_model = concatenate([channel_model.output, coord_model.output])
        z = Dense(Nt,activation="linear")(combined_model)
        model = Model(inputs=[channel_model.input,coord_model.input],outputs=z)
        model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
        model.summary()
        hist = model.fit([X_channel_train,X_coord_train],y_train, 
            validation_split=0.2,
            epochs=num_epochs,batch_size=batch_size,
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
    elif 'coord' in args.input and 'lidar' in args.input:
        combined_model = concatenate([coord_model.output,lidar_model.output])
        z = Dense(Nt,activation="linear")(combined_model)
        model = Model(inputs=[coord_model.input,lidar_model.input],outputs=z)
        model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
        model.summary()
        hist = model.fit([X_coord_train,X_lidar_train],y_train, 
            validation_split=0.2,
            epochs=num_epochs,batch_size=batch_size,
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

    elif 'coord' in args.input and 'img' in args.input:
        combined_model = concatenate([coord_model.output,img_model.output])
        z = Dense(Nt,activation="linear")(combined_model)
        model = Model(inputs=[coord_model.input,img_model.input],outputs=z)
        model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
        model.summary()
        hist = model.fit([X_coord_train,X_img_train],y_train,
            validation_split=0.2,
            epochs=num_epochs,batch_size=batch_size,
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
    
    else:
        combined_model = concatenate([lidar_model.output,img_model.output])
        z = Dense(Nt,activation="linear")(combined_model)
        model = Model(inputs=[lidar_model.input,img_model.input],outputs=z)
        model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
        model.summary()
        hist = model.fit([X_lidar_train,X_img_train],y_train, 
            validation_split=0.2,
            epochs=num_epochs,batch_size=batch_size,
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
elif multimodal == 3:
    if 'channel' in args.input and 'coord' in args.input and 'img' in args.input:
        combined_model = concatenate([channel_model.output,coord_model.output, img_model.output])
        z = Dense(Nt,activation="linear")(combined_model)
        model = Model(inputs=[channel_model.input,coord_model.input, img_model.input],outputs=z)
        model.compile(loss="mse",
                        optimizer=opt
                        #metrics=[metrics.categorical_accuracy,
                                #metrics.top_k_categorical_accuracy,
                                #top_50_accuracy]
                                )
        model.summary()
        hist = model.fit([X_channel_train,X_coord_train,X_img_train],y_train,
                validation_split=0.2,
                epochs=num_epochs,batch_size=batch_size,
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
    if 'channel' in args.input and 'coord' in args.input and 'lidar' in args.input:
        combined_model = concatenate([channel_model.output,coord_model.output, lidar_model.output])
        z = Dense(Nt,activation="linear")(combined_model)
        model = Model(inputs=[channel_model.input,coord_model.input, lidar_model.input],outputs=z)
        model.compile(loss="mse",
                        optimizer=opt
                        #metrics=[metrics.categorical_accuracy,
                                #metrics.top_k_categorical_accuracy,
                                #top_50_accuracy]
                                )
        model.summary()
        hist = model.fit([X_channel_train,X_coord_train,X_lidar_train],y_train,
                validation_split=0.2,
                epochs=num_epochs,batch_size=batch_size,
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
    if 'channel' in args.input and 'lidar' in args.input and 'img' in args.input:
        combined_model = concatenate([channel_model.output,lidar_model.output, img_model.output])
        z = Dense(Nt,activation="linear")(combined_model)
        model = Model(inputs=[channel_model.input,lidar_model.input, img_model.input],outputs=z)
        model.compile(loss="mse",
                        optimizer=opt
                        #metrics=[metrics.categorical_accuracy,
                                #metrics.top_k_categorical_accuracy,
                                #top_50_accuracy]
                                )
        model.summary()
        hist = model.fit([X_channel_train,X_lidar_train,X_img_train],y_train,
                validation_split=0.2,
                epochs=num_epochs,batch_size=batch_size,
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
    if 'lidar' in args.input and 'coord' in args.input and 'img' in args.input:
        combined_model = concatenate([lidar_model.output,coord_model.output, img_model.output])
        z = Dense(Nt,activation="linear")(combined_model)
        model = Model(inputs=[lidar_model.input,coord_model.input, img_model.input],outputs=z)
        model.compile(loss="mse",
                        optimizer=opt
                        #metrics=[metrics.categorical_accuracy,
                                #metrics.top_k_categorical_accuracy,
                                #top_50_accuracy]
                                )
        model.summary()
        hist = model.fit([X_lidar_train,X_coord_train,X_img_train],y_train,
                validation_split=0.2,
                epochs=num_epochs,batch_size=batch_size,
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
elif multimodal == 4:
    combined_model = concatenate([channel_model.output,lidar_model.output,coord_model.output, img_model.output])
    z = Dense(Nt,activation="linear")(combined_model)
    model = Model(inputs=[channel_model.input,lidar_model.input,coord_model.input, img_model.input],outputs=z)
    model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
    model.summary()
    hist = model.fit([X_channel_train,X_lidar_train,X_coord_train,X_img_train],y_train,
            validation_split=0.2,
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
else:
    if 'coord' in args.input:
        model = coord_model
        model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
        model.summary()
        hist = model.fit(X_coord_train,y_train, 
            validation_split=0.2,
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

    elif 'img' in args.input:
        model = img_model  
        model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
        model.summary()
        hist = model.fit(X_img_train,y_train, 
            validation_split=0.2,
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

    elif 'channel':
        model = channel_model
        model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
        model.summary()
        hist = model.fit(X_train,y_train, 
            validation_split=0.2,
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
    else:
        model = lidar_model
        model.compile(loss="mse",
                    optimizer=opt
                    #metrics=[metrics.categorical_accuracy,
                            #metrics.top_k_categorical_accuracy,
                            #top_50_accuracy]
                            )
        model.summary()
        hist = model.fit(X_lidar_train,y_train, 
            validation_split=0.2,
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