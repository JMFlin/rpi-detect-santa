# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os





if __name__ == '__main__':


    direc = os.getcwd() + "\\images/not_santa/" 

    for filename in os.listdir(direc):
        print(filename)
        imagePath = direc + filename
        print(imagePath)

        image = cv2.imread(imagePath)

    #image = cv2.imread("C:\Users\janne.m.flinck\Desktop/test\images/santa/00000484.jpg")
    orig = image.copy()
     
    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)


    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model("model")
     
    # classify the input image
    (notSanta, santa) = model.predict(image)[0]

    # build the label
    label = "Santa" if santa > notSanta else "Not Santa"
    proba = santa if santa > notSanta else notSanta
    label = "{}: {:.2f}%".format(label, proba * 100)
     
    # draw the label on the image
    #output = imutils.resize(orig, width=400)
    output = orig
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2)
     
    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)