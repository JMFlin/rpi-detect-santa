# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from networks.lenet.lenet import LeNet
from networks.convnet3.convnet3 import ConvNet3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os



def load_images(directory_positive, directory_negative):

    data = []
    labels = []

    for filename in os.listdir(directory_positive):

        imagePath = directory_positive + filename
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGTH))
        image = img_to_array(image)
        data.append(image)

        #label = "santa"

        label = 1 #if label == "santa" else 0
        labels.append(label)
 
    for filename in os.listdir(directory_negative):

        imagePath = directory_negative + filename

        image = cv2.imread(imagePath)
        
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGTH))
        image = img_to_array(image)
        data.append(image)

        #label = "not_santa"#imagePath.split(os.path.sep)[-1]
        label = 0 #if label == "santa" else 0
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return(data, labels)

def plot_training(epochs, history, plot_directory):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()

    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_directory + "training-loss-and-accuracy.jpg")


def create_train_test(data, labels):
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.25, random_state=42)
     
    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    return(trainX, testX, trainY, testY)


if __name__ == '__main__':
    
    # initialize the number of epochs to train for, initial learning rate,
    # and batch size
    EPOCHS = 25
    INIT_LR = 1e-3
    BS = 32
    IMAGE_WIDTH = 28
    IMAGE_HEIGTH = 28
    directory_positive = os.getcwd() + "/images/positive/" 
    directory_negative = os.getcwd() + "/images/negative/"
    plot_directory = os.getcwd() + "/networks/convnet3/training-plots/"
    model_directory = os.getcwd() + "/networks/convnet3/model/"
     
    # initialize the data and labels
    print("[INFO] loading images...")
    data, labels = load_images(directory_positive, directory_negative)

    print("[INFO] splitting into training and testing...")
    (trainX, testX, trainY, testY) = create_train_test(data, labels)

    # initialize the model
    print("[INFO] building and compiling model...")
    model = ConvNet3.build(width=IMAGE_WIDTH, height=IMAGE_HEIGTH, depth=3, classes=2)
    model = ConvNet3.compile(model=model, lr=INIT_LR, decay=INIT_LR / EPOCHS, metrics = "accuracy")

    # train the network
    print("[INFO] training network...")
    (model, history) = ConvNet3.train(model, trainX, testX, trainY, testY, BS, EPOCHS)

    ###
    #print(model.evaluate(testX, testY, batch_size=128))
    ###Scalar test loss

    print("[INFO] creating validation plot...")
    plot_training(EPOCHS, history, plot_directory)
     
    # save the model to disk
    print("[INFO] serializing network...")
    model.save(model_directory + "model.h5")

    