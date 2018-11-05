from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from keras.layers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np

from keras import backend as K

class ModelLoader():

    def __init__(self, model_name, input_shape, optimizer=None, ):

        self.model_name = model_name
        self.input_shape = input_shape

        # Loads the specified model
        if self.model_name == "multiple_output":
            print('Loading multiple_output model')

            self.model = self.multiple_output()

        elif self.model_name == "simple":
            print('Loading simple model')

            self.model = self.simple()

        else:
            raise Exception('No model with name {} found!'.format(model_name))
        # Define metrics

        metrics = ['mse']
        # If no optimizer is given, use Adam as default
        if optimizer is None:
            optimizer = Adam()

        self.model.compile(loss='mean_squared_error',
                           optimizer=optimizer,
                           metrics=metrics)
        
        print(self.model.summary())
    
    def normalization(self, input_tensor):
        output = Lambda(lambda x: x/255.0 - 0.5, input_shape=self.input_shape)(input_tensor)
        output = Cropping2D(cropping=((70, 25), (0,0)))(output)
        return output

    def multiple_output(self):
        kernel_size = (3, 3)
        pool_size = (2, 2)
        dense_units = 128

        input_img = Input(shape=self.input_shape)

        # preprocessing
        angle = self.normalization(input_img)
        
        angle = Conv2D(32, kernel_size, activation='relu', padding='same')(input_img)
        angle = MaxPooling2D(pool_size=pool_size)(angle)
        angle = Dropout(0.25)(angle)

        angle = Conv2D(64, kernel_size, activation='relu', padding='same')(angle)
        angle = MaxPooling2D(pool_size=pool_size)(angle)
        angle = Dropout(0.25)(angle)
                            
        angle = Conv2D(128, kernel_size, activation='relu', padding='same')(angle)
        angle = MaxPooling2D(pool_size=pool_size)(angle)
        angle = Dropout(0.25)(angle)

        angle = Flatten()(angle)
        angle = Dense(128, init='normal', activation='relu')(angle)
        angle = Dropout(0.25)(angle)
        angle = Dense(1, init='normal', activation='linear')(angle)

        model = Model(inputs=input_img, outputs=angle)
        return model

    def simple(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(2))

        return model
