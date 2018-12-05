from keras.preprocessing.image import img_to_array
from keras.models import load_model
#from gpiozero import LEDBoard
#from gpiozero.tools import random_values
from imutils.video import VideoStream
from threading import Thread
import imutils
import cv2
import time
from time import sleep
import os
import numpy as np
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

#led_pins = [2, 3, 4, 17]
led_pins = [2]
# define the paths to the Not Santa Keras deep learning model and
# audio file
MODEL_PATH = "model"
AUDIO_PATH = "~/Desktop/song.mp3"

# initialize the total number of frames that *consecutively* contain
# santa along with threshold required to trigger the santa alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 20

# initialize is the santa alarm has been triggered
SANTA = False


# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)
 
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
sleep(2.0)

ACTIVATION = 0

def play_christmas_music(p):
	# construct the command to play the music, then execute the
	# command
	try:
		command = "mpg123 {}".format(p)
		os.system(command)
	except KeyboardInterrupt:
		pass
	
def activate_leds(led_pins):
	try:
		for i in led_pins:
			GPIO.setup(i, GPIO.OUT)
			GPIO.output(i, GPIO.LOW)
			print("Led", i, "is on")
			sleep(4)
			
			#### Comment this when ready to use properly
			GPIO.output(i, GPIO.HIGH)
			#### Comment this when ready to use properly
			
			sleep(2)
	except KeyboardInterrupt:
		pass
 
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	
	# prepare the image to be classified by our deep learning network
	image = cv2.resize(frame, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
 
	# classify the input image and initialize the label and
	# probability of the prediction
	(santa, notSanta) = model.predict(image)[0]
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
		TOTAL_CONSEC = TOTAL_CONSEC + 1
		# check to see if we should raise the santa alarm
		#print(TOTAL_CONSEC)
		#print(not SANTA)
		
		#not SANTA
		if not SANTA and TOTAL_CONSEC >= TOTAL_THRESH and ACTIVATION = 0:
			print("[INFO] prediction made...")
			# indicate that santa has been found
			SANTA = True
			
			print("[INFO] activating led...")
			# light up the christmas lights
			ledThread = Thread(target=activate_leds, args=(led_pins,))
			ledThread.daemon = True
			ledThread.start()
			
			print("[INFO] playing music...")
			# play some christmas tunes
			musicThread = Thread(target=play_christmas_music,
			args=(AUDIO_PATH,))
			musicThread.daemon = False
			musicThread.start()
			
			ACTIVATION = ACTIVATION + 1
			
	# otherwise, reset the total number of consecutive frames and the
	# santa alarm
	else:
		TOTAL_CONSEC = 0
		ACTIVATION = 0
		SANTA = False
		for i in led_pins:
			GPIO.output(i, GPIO.HIGH)
			#sleep(0.5)
			
	# build the label and draw it on the frame
	label = "{}: {:.2f}%".format(label, proba * 100)
	frame = cv2.putText(frame, label, (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break	
			

try:
	GPIO.cleanup()
except:
	pass
 
# initialize the christmas tree
