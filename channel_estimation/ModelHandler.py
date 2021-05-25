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
from cGANDiscriminator import Discriminator
from cGANGenerator import Generator

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
        
        if(model_type == 'inception_single'):
            input_inc = Input(shape = input_shape)

            tower_1 = Conv2D(4, (1,1), padding='same', activation='tanh')(input_inc)
            tower_1 = Conv2D(8, (2,output_dim[1]), padding='same', activation='tanh')(tower_1)
            tower_1 = Conv2D(16, (3,output_dim[1]), padding='same', activation='tanh')(tower_1)
            tower_2 = Conv2D(4, (1,1), padding='same', activation='tanh')(input_inc)
            tower_2 = Conv2D(16, (3,output_dim[1]), padding='same', activation='tanh')(tower_2)
            tower_2 = Conv2D(16, (5,output_dim[1]), padding='same', activation='tanh')(tower_2)
            tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_inc)
            tower_3 = Conv2D(4, (1,1), padding='same', activation='tanh')(tower_3)
            
            output = concatenate([tower_1, tower_2, tower_3], axis = 3)
            
            if(chain=='segment'):
                model = output
                
            else:
                output = Dropout(0.25)(output)
                output = Flatten()(output)
                output = Dense(num_classes,activation='linear')(output)
                output = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(output)
                out = Reshape(output_dim)(output)
                
                model = Model(inputs = input_inc, outputs = out)

        elif(model_type == 'light_image'):
            input_inc = Input(shape = input_shape)

            tower_1 = Conv2D(8, (3,3), padding='same', activation='relu')(input_inc)
            tower_1 = MaxPooling2D((2,2), strides=(1,1), padding='same')(tower_1)
            tower_1 = Conv2D(8, (3,3), padding='same', activation='relu')(tower_1)
            tower_1 = MaxPooling2D((2,2), strides=(1,1), padding='same')(tower_1)
            tower_1 = Conv2D(8, (3,3), padding='same', activation='relu')(tower_1)
            #output = concatenate([tower_1, tower_2, tower_3], axis = 3)
            output = tower_1

            output = Dropout(0.25)(output)
            output = Flatten()(output)
            output = Dense(num_classes,activation='linear')(output)
            output = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(output)
            out = Reshape(output_dim)(output)
                
            model = Model(inputs = input_inc, outputs = out)

        
        elif(model_type == 'coord_mlp'):
            input_coord = Input(shape = input_shape)
            
            layer = Dense(4,activation='relu')(input_coord)
            layer = Dense(16,activation='relu')(layer)
            layer = Dense(64,activation='relu')(layer)
            output = Dense(num_classes,activation='linear')(layer)
            output = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(output)
            out = Reshape(output_dim)(output)
            
            
            model = Model(inputs = input_coord, outputs = out)
            
        elif(model_type == 'lidar_marcus'):
            dropProb=0.3
            input_lid = Input(shape = input_shape)
                        
            layer = Conv2D(10,kernel_size=(13,13),
                                activation='tanh',
                                padding="SAME",
                                input_shape=input_shape)(input_lid)
            layer = Conv2D(30, (11, 11), padding="SAME", activation='tanh')(layer)
            layer = Conv2D(25, (9, 9), padding="SAME", activation='tanh')(layer)
            layer = MaxPooling2D(pool_size=(2, 1))(layer)
            layer = Dropout(dropProb)(layer)
            layer = Conv2D(20, (7, 7), padding="SAME", activation='tanh')(layer)
            layer = MaxPooling2D(pool_size=(1, 2))(layer)
            layer = Conv2D(15, (5, 5), padding="SAME", activation='tanh')(layer)
            layer = Dropout(dropProb)(layer)
            layer = Conv2D(10, (3, 3), padding="SAME", activation='tanh')(layer)
            layer = Conv2D(1, (1, 1), padding="SAME", activation='tanh')(layer)
            layer = Flatten()(layer)
            output = Dense(num_classes,activation='linear')(layer)
            output = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(output)
            out = Reshape(output_dim)(output)
            
            model = Model(inputs = input_lid, outputs = out)
        
        elif(model_type == 'channel_conv'):
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
        
        elif(model_type == 'channel_dense'):
            dropProb=0.3
            input_channel = Input(shape = input_shape)

            N = 500
                    
            layer = Flatten()(input_channel)
            layer = Dense(N, activation='tanh')(layer)
            layer = Dense(N, activation='tanh')(layer)
            layer = Dropout(dropProb)(layer)

            layer = Flatten()(layer)
            layer = Dense(num_classes,activation='linear')(layer)
            layer = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(layer)
            out = Reshape(output_dim)(layer)
            
            architecture = Model(inputs = input_channel, outputs = out)
        
        elif(model_type == 'splited_y'):
            dropProb=0.3
                        
            N = 150
            
            input_real = Input(shape = input_shape)
            real_layer = Flatten()(input_real)
            real_layer = Dense(N, activation='tanh')(real_layer)
            real_layer = Dense(N, activation='tanh')(real_layer)
            real_layer = Dropout(dropProb)(real_layer)

            real_layer = Flatten()(real_layer)
            real_layer = Dense(num_classes,activation='linear')(real_layer)
            real_layer = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(real_layer)
            real_out = Reshape(output_dim)(real_layer)
            
            real_arch = Model(inputs = input_real, outputs = real_out)

            input_img = Input(shape = input_shape)
            img_layer = Flatten()(input_img)
            img_layer = Dense(N, activation='tanh')(img_layer)
            img_layer = Dense(N, activation='tanh')(img_layer)
            img_layer = Dropout(dropProb)(img_layer)

            img_layer = Flatten()(img_layer)
            img_layer = Dense(num_classes,activation='linear')(img_layer)
            img_layer = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(img_layer)
            img_out = Reshape(output_dim)(img_layer)
            
            img_arch = Model(inputs = input_img, outputs = img_out) 

            combined_model = concatenate([real_arch.output, img_arch.output])
            z = Dense(16,activation="linear")(combined_model)
            
            model = Model(inputs=[real_arch.input,img_arch.input],outputs=z)
            
        return model


#zhang 2020
#model.add(tf.keras.layers.Dense(8192, activation='relu', input_shape=input_shape[1:]))
#model.add(tf.keras.layers.Dense(8192, activation='relu'))
#model.add(tf.keras.layers.Dense(128, activation='relu'))

#dong 2020
