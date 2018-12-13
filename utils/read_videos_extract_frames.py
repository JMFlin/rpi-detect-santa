import cv2 
import os
from datetime import date, timedelta, datetime
import sys
import math
import random
import re
#import cv2
import argparse


def read_videos_extract_frames(args):

    picture_count = 0 
    now = datetime.now()

    path_to_videos = args["path"]
    save_folder = args["save"]

    directory = os.fsencode(path_to_videos)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".MOV") or filename.endswith(".mp4"): 
            cap = cv2.VideoCapture(path_to_videos + filename)   # capturing the video from the given path
            frameRate = cap.get(5) #frame rate
            while(cap.isOpened()):
                frameId = cap.get(1) #current frame number
                ret, frame = cap.read()
                
                if (ret != True):
                    break
                if (frameId % math.floor(frameRate) == 0):

                    filename ="C:/Users/janne.m.flinck/Desktop/rpi-detect-santa/images/stage/" + "pic_" + now.strftime("%Y-%m-%d-%H-%M") + "_frame%d.jpg" % picture_count;picture_count+=1
                    print(filename)
                    cv2.imwrite(filename, frame)
            cap.release()


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--path", required=True,
        help="path to video folder")
    ap.add_argument("-o", "--save", required=True,
        help="path to directory where images should be saved")
    args = vars(ap.parse_args())

    read_videos_extract_frames(args)
    #read_videos_extract_frames("C:/Users/janne.m.flinck/Desktop/Neste/image_recognition/videos/train/not_safe/",  save_folder = 'train/not_safe/')
    #read_videos_extract_frames("C:/Users/janne.m.flinck/Desktop/Neste/image_recognition/videos/train/safe/",  save_folder = 'train/safe/')

    #read_videos_extract_frames("C:/Users/janne.m.flinck/Desktop/Neste/image_recognition/videos/test/not_safe/",  save_folder = 'test/not_safe/')
    #read_videos_extract_frames("C:/Users/janne.m.flinck/Desktop/Neste/image_recognition/videos/test/safe/",  save_folder = 'test/safe/')
    