from flask import Flask, jsonify
import pyaudio
import wave
from pyAudioAnalysis import audioTrainTest as aT
import numpy as np
import sys
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "recording6.wav"
 
audio = pyaudio.PyAudio()

frames = []

def record():
	# start Recording
	stream = audio.open(format=FORMAT, channels=CHANNELS,
		                rate=RATE, input=True,
		                frames_per_buffer=CHUNK)
	# print "recording..."
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)
	# print "finished recording"
	# stop Recording
	stream.stop_stream()
	stream.close()
	audio.terminate()
	return 'recorded'

def save(output_name):
	waveFile = wave.open(output_name, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(frames))
	waveFile.close()

def classify(filename):
	isSignificant = 0.8 #TN/FP Threshold

	# Result, P, classNames = aT.fileClassification(filename, "knnDE","knn")
	Result, P, classNames = aT.fileClassification(filename, "knnDE2","knn")
	winner = np.argmax(P) #pick the result with the highest probability value.
	ans = {}
	if P[winner] > isSignificant :
	 	# print("File: " + filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner]))
	 	# ans += "File: " + filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner])
	 	ans = {"class": classNames[winner], "prob":P[winner]}
	else :
	  # print("Can't classify sound: " + str(P))
	 	ans = {"class": "none detected"}

	return ans



app = Flask(__name__)

@app.route('/')
def hello_world():
	record()
	save(WAVE_OUTPUT_FILENAME)
	ans = classify(WAVE_OUTPUT_FILENAME)
 	return "Hello, World!\n" + ans['class']

@app.route('/api/get_messages', methods = ['GET'])
def get_messages():
    # json = request.get_json()
    # if json['user'] == "larry":
    #     return jsonify({'messages':['test1', 'test2']})
    record()
    save(WAVE_OUTPUT_FILENAME)
    ans = classify(WAVE_OUTPUT_FILENAME)
    return jsonify(ans)
