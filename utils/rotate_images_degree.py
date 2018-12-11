import cv2 
import os
import sys
from PIL import Image

def flip_images_right(path_to_pic, degree):

    directory = os.fsencode(path_to_pic)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        img = Image.open(path_to_pic + filename)   # capturing the video from the given path
        img.rotate(degree, expand=True).save(path_to_pic + filename)

if __name__ == '__main__':
    
    flip_images_right("C:/Users/janne.m.flinck/Desktop/rpi-detect-santa/images/stage/", degree = 270)
