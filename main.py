import RPi.GPIO as GPIO
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from threading import Thread, Event
import imutils
import cv2
import time
from time import sleep
import os
import numpy as np

GPIO.setmode(GPIO.BCM)
MODEL_PATH = "networks/lenet/models/model-64-64.h5"
AUDIO_PATH = "songs/jingle_bell_rock.mp3"
TOTAL_THRESH_SANTA = 50
TOTAL_THRESH_NOT_SANTA = 100
TOTAL_CONSEC_SANTA = 0
TOTAL_CONSEC_NOT_SANTA = 0

led_pins = [2, 3]

print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
sleep(2.0)

#body_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_fullbody.xml')

SANTA = False

print("[INFO] loading model...")
model = load_model(MODEL_PATH)

def play_christmas_music(path):
    # construct the command to play the music, then execute the
    # command
    try:
        command = "mpg123 {}".format(path)
        os.system(command)
    except KeyboardInterrupt:
        pass


def activate_leds(led_pins):

    try:
        for i in led_pins:
            GPIO.setup(i, GPIO.OUT)
            GPIO.output(i, GPIO.LOW)
            print("[INFO] Led", i, "is on")
            sleep(2)
            GPIO.output(i, GPIO.HIGH)
            print("[INFO] Led", i, "is off")
        sleep(3)
        for i in led_pins:
            GPIO.setup(i, GPIO.OUT)
            GPIO.output(i, GPIO.LOW)
            print("[INFO] Led", i, "is on")
        print("[INFO] Both leds are on")
    except KeyboardInterrupt:
        pass


def deactivate_leds(led_pins):
    try:
        for i in led_pins:
            GPIO.output(i, GPIO.HIGH)
        print("[INFO] Both leds are off")
    except KeyboardInterrupt:
        pass


def stop_christmas_music(path):
    # construct the command to stop the music, then execute the
    # command
    try:
        command = "pkill mpg123"  # "pidof mpg123 | xargs kill -9"
        os.system(command)
    except KeyboardInterrupt:
        pass


def frame_image(vs):

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, height=1200, width=1200)

    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (64, 64))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return(frame, image)


def start_threading_leds(led_pins):
    print("[INFO] activating led...")
    ledThread = Thread(target=activate_leds, args=(led_pins,))
    ledThread.daemon = True
    ledThread.start()
    return


def start_threading_music(AUDIO_PATH):
    print("[INFO] playing music...")
    musicThread = Thread(target=play_christmas_music,
                         args=(AUDIO_PATH,))
    musicThread.daemon = False
    musicThread.start()
    return


def activate_detection(
        TOTAL_THRESH_SANTA,
        TOTAL_THRESH_NOT_SANTA,
        model,
        led_pins):

    # loop over the frames from the video stream
    while True:
        ACTIVATION = 0

        frame, image = frame_image(vs)

        # classify the input image and initialize the label and
        # probability of the prediction
        (notSanta, santa) = model.predict(image)[0]
        label = "Not Santa"
        proba = notSanta

        # check to see if santa was detected using our convolutional
        # neural network
        #print("[INFO] prediction made...")
        if santa > notSanta:

            # update the label and prediction probability
            label = "Santa"
            proba = santa

            # increment the total number of consecutive frames that
            # contain santa
            TOTAL_CONSEC_SANTA = TOTAL_CONSEC_SANTA + 1

            # check to see if we should raise the santa alarm
            if not SANTA and TOTAL_CONSEC_SANTA >= TOTAL_THRESH_SANTA and ACTIVATION == 0:
                print("[INFO] prediction made...")
                # indicate that santa has been found
                SANTA = True

                start_threading_leds(led_pins)

                start_threading_music(AUDIO_PATH)

                ACTIVATION = ACTIVATION + 1

            # build the label and draw it on the frame
            label = "{}: {:.2f}%".format(
                "Santa found with probability", proba * 100)
            frame = cv2.putText(frame, label, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:

            TOTAL_CONSEC_NOT_SANTA = TOTAL_CONSEC_NOT_SANTA + 1
            # build the label and draw it on the frame
            label = "{}: {:.2f}%".format(
                "Trigger probability", (1 - proba) * 100)  # label
            frame = cv2.putText(frame, label, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # otherwise, reset the total number of consecutive frames and the
        # santa alarm
        if TOTAL_CONSEC_NOT_SANTA >= TOTAL_THRESH_NOT_SANTA:
            TOTAL_CONSEC_NOT_SANTA = 0
            TOTAL_CONSEC_SANTA = 0
            ACTIVATION = 0
            SANTA = False
            try:
                stop_christmas_music(AUDIO_PATH)
                deactivate_leds(led_pins)
            except BaseException:
                pass

        cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)

        cv2.resizeWindow("Frame", 1200, 1200)
        #cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #body = body_cascade.detectMultiScale(frame)
        # for (x,y,w,h) in body:
        #	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    return


def clean_up(AUDIO_PATH):
    try:
        GPIO.cleanup()
        stop_christmas_music(AUDIO_PATH)
    except BaseException:
        pass


activate_detection(TOTAL_THRESH_SANTA=TOTAL_THRESH_SANTA,
                   TOTAL_THRESH_NOT_SANTA=TOTAL_THRESH_NOT_SANTA,
                   model=model,
                   led_pins=led_pins)

clean_up(AUDIO_PATH=AUDIO_PATH)
