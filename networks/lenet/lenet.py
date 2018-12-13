# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K
 
class LeNet:
    
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
 
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(96, 3, 3, border_mode='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
 
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
 
        return model

    @staticmethod
    def compile(model, lr, decay, metrics):
        opt = Adam(lr=lr, decay=decay)
        model.compile(loss="binary_crossentropy", optimizer=opt,
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