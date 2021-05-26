#Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of the UFPA - ITU Artificial Intelligence/Machine Learning in 5G Challenge (http://ai5gchallenge.ufpa.br/).
#Author       		: Ailton Oliveira, Aldebaro Klautau, Arthur Nascimento, Diego Gomes, Jamelly Ferreira, Walter Frazão
#Email          	: ml5gphy@gmail.com                                          
#License		: This script is distributed under "Public Domain" license.
###################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2D, add,\
    Flatten, MaxPooling2D, Dense, Reshape, Input, Dropout, concatenate, Conv1D, Lambda
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.utils
from tensorflow.keras import backend as K
import numpy as np
import copy

class ModelHandler:
    
    
    def createArchitecture(self,model_type,num_classes,input_shape,chain, output_dim=None):
        '''
        Returns a NN model.
        modelType: a string which defines the structure of the model
        numClasses: a scalar which denotes the number of classes to be predicted
        input_shape: a tuple with the dimensions of the input of the model
        chain: a string which indicates if must be returned the complete model
        up to prediction layer, or a segment of the model.
        output_shape = a tuple with the dimensions of the output of the model 
        '''

        H_normalization_factor = np.sqrt(output_dim[0]//2 * output_dim[1])
        
        dropProb=0.3
        input_channel = Input(shape = input_shape)
                    
        layer = Conv1D(64, 3, activation='tanh', padding="SAME")(input_channel)
        layer = Conv1D(64, 3, padding="SAME", activation='tanh')(layer)
        layer = Dropout(dropProb)(layer)

        layer = Flatten()(layer)
        layer = Dense(num_classes,activation='linear')(layer)
        layer = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(layer)
        out = Reshape(output_dim)(layer)
        
        model = Model(inputs = input_channel, outputs = out)       
              
        return model



