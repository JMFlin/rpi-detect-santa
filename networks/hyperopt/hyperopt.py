# import the necessary packages
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
 
class HyperOpt:
    #https://github.com/maxpumperla/hyperas/blob/master/examples/cifar_generator_cnn.py
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
         model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout({{uniform(0, 1)}}))

        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout({{uniform(0, 1)}}))

        # If we choose 'four', add an additional fourth layer
        if {{choice(['three', 'four'])}} == 'four':
            model.add(Conv2D(50, (5, 5), padding="same"))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

        result = model.fit(x_train, y_train,
                  batch_size={{choice([64, 128])}},
                  epochs=2,
                  verbose=2,
                  validation_split=0.1)
        #get the highest validation accuracy of the training epochs
        validation_acc = np.amax(result.history['val_acc']) 
        print('Best validation acc of epoch:', validation_acc)
        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


    @staticmethod
    def compile(model, lr, decay, metrics):
        
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
        return(model)

    @staticmethod
    def train(model, trainX, testX, trainY, testY, BS, EPOCHS):

        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode="nearest")
        
        history = model.fit_generator(aug.flow(trainX, trainY, batch_size={{choice([64, 128])}}),
            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
            epochs=EPOCHS, verbose=1)

        return(model, history)