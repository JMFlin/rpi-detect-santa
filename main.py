from keras.preprocessing.image import img_to_array
from keras.models import load_model
#from gpiozero import LEDBoard
#from gpiozero.tools import random_values
from imutils.video import VideoStream
from threading import Thread, Event
import imutils
import cv2
import time
from time import sleep
import os
import numpy as np
import RPi.GPIO as GPIO

def main():

	GPIO.setmode(GPIO.BCM)

	#led_pins = [2, 3, 4, 17]
	led_pins = [2, 3]
	# define the paths to the Not Santa Keras deep learning model and
	# audio file
	MODEL_PATH = "networks/lenet/models/model"
	AUDIO_PATH = "songs/jingle_bell_rock.mp3"

	# initialize the total number of frames that *consecutively* contain
	# santa along with threshold required to trigger the santa alarm

	TOTAL_CONSEC_SANTA = 0
	TOTAL_CONSEC_NOT_SANTA = 0
	TOTAL_THRESH_SANTA = 50
	TOTAL_THRESH_NOT_SANTA = 100

	# load the model
	print("[INFO] loading model...")
	model = load_model(MODEL_PATH)
	 
	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	#vs = VideoStream(src=0).start()
	vs = VideoStream(usePiCamera=True).start()
	sleep(2.0)

	activate_detection(TOTAL_CONSEC_SANTA = TOTAL_CONSEC_SANTA, 
		TOTAL_CONSEC_NOT_SANTA = TOTAL_CONSEC_NOT_SANTA, 
		TOTAL_THRESH_SANTA = TOTAL_THRESH_SANTA, 
		model = model)
		
	clean_up(AUDIO_PATH = AUDIO_PATH)



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
		command = "pkill mpg123" #"pidof mpg123 | xargs kill -9"
		os.system(command)
	except KeyboardInterrupt:
		pass

def frame_image():
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	
	# prepare the image to be classified by our deep learning network
	image = cv2.resize(frame, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	
	return(frame, image)

		
def activate_detection(TOTAL_CONSEC_SANTA, TOTAL_CONSEC_NOT_SANTA, TOTAL_THRESH_SANTA, model):
	# initialize is the santa alarm has been triggered
	SANTA = False
	# loop over the frames from the video stream
	while True:
		ACTIVATION = 0
		
		frame, image = frame_image()

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
			TOTAL_CONSEC_SANTA = TOTAL_CONSEC_SANTA + 1
			
			# check to see if we should raise the santa alarm
			if not SANTA and TOTAL_CONSEC_SANTA >= TOTAL_THRESH_SANTA and ACTIVATION == 0:
				print("[INFO] prediction made...")
				# indicate that santa has been found
				SANTA = True
				
				print("[INFO] activating led...")
				ledThread = Thread(target=activate_leds, args=(led_pins,))
				ledThread.daemon = True
				ledThread.start()
								
				print("[INFO] playing music...")
				musicThread = Thread(target=play_christmas_music,
				args=(AUDIO_PATH,))
				musicThread.daemon = False
				musicThread.start()
				
				ACTIVATION = ACTIVATION + 1
		else:
			
			TOTAL_CONSEC_NOT_SANTA = TOTAL_CONSEC_NOT_SANTA + 1	
		
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
			except:
				pass
		
		show_output(label, proba, frame)

	return

def show_output(label, proba, frame):
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

	return
			


 
def clean_up(AUDIO_PATH):
	try:
		GPIO.cleanup()
		stop_christmas_music(AUDIO_PATH)
	except:
		pass

main()
