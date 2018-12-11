# import the necessary packages
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
 
class ConvNet3:
    
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
 
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Dense(512, input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('sigmoid'))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation('sigmoid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dropout(0.1))
        model.add(Dense(classes))
        model.add(Activation('sigmoid'))

        return model

    @staticmethod
    def compile(model, lr, decay, metrics):
        
        opt = "rmsprop"
        model.compile(loss="binary_crossentropy", 
            optimizer=opt,
            metrics=[metrics])
        return(model)

    @staticmethod
    def train(model, trainX, testX, trainY, testY, BS, EPOCHS):

        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode="nearest")
        
        history = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
            epochs=EPOCHS, verbose=1)
        return(model, history)