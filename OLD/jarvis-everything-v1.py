from halo import Halo
import time, logging
import argparse
import os
import glob
import struct
import sys
from datetime import datetime
from threading import Thread
import deepspeech
from audio_tools import VADAudio
import numpy as np
import pyaudio
import soundfile
from pvporcupine import *

import pyttsx3
import random
import requests

import aiml
import time
import socket

import wordifynum
from wakeonlan import send_magic_packet

logging.basicConfig(level=logging.INFO)

def jprint(str):
    print(f"[J.A.R.V.I.S] {str}")

jprint('Loading...')

engine=pyttsx3.init('espeak')
voices=engine.getProperty('voices')
engine.setProperty('voice', 'english')

kernel = aiml.Kernel()
kernel.bootstrap(learnFiles="std-startup.xml", commands="load aiml b")

def speak(text):
    print(f"JARVIS: {text}")
    engine.say(text)
    engine.runAndWait()

def exist(query, terms):
    for term in terms:
        if term in query:
            return True
    return False

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

known_computers = {
    'danny': '30:9C:23:DF:E2:B3'
}

def getResponse(query):
    if query == "":
        return "I didnt catch that."
    elif "cancel" in query:
        return "Cancelled"
    elif "morning" in query:
        return "Good morning, sir"
    elif "afternoon" in query:
        return "Good afternoon, sir"
    elif "are you up" in query:
        return "For you sir, always"
    elif "turn on" in query:
        if exist(query, ["danny's computer", "my computer","my piece"]):
            send_magic_packet(known_computers['danny'])
            return "Your computer should be turning on."
        elif "the lights" in query:    
            return "Program that"
    elif "turn off" in query:
        if exist(query, ["danny's computer", "my computer","my piece"]):
            send_magic_packet(known_computers['danny'])
            send_magic_packet(known_computers['danny'])
            send_magic_packet(known_computers['danny'])
            send_magic_packet(known_computers['danny'])
            send_magic_packet(known_computers['danny'])
            send_magic_packet(known_computers['danny'])
            send_magic_packet(known_computers['danny'])
            return "Your computer should be turning off."
    elif "network address" in query:
        ip = get_ip().split('.')
        ip = map(wordifynum.say_ipnumber, ip)
        return f"My local network address is {', dot '.join(ip)}"
    elif "i am home" in query:
        return "Welcome home, sir. I have gone ahead and turned on the lights"
    elif "learn" in query:
        if "skill" in query:
            return "What would you like me to learn"
    return "Hello sir"
    #return kernel.respond(query)


class PorcupineDemo(Thread):
    def __init__(self, library_path, model_path, keyword_paths, sensitivities, input_device_index=None, output_path=None):
        super(PorcupineDemo, self).__init__()

        self._library_path = library_path
        self._model_path = model_path
        self._keyword_paths = keyword_paths
        self._sensitivities = sensitivities
        self._input_device_index = input_device_index

        self._output_path = output_path
        if self._output_path is not None:
            self._recorded_frames = []
            
        #Load DeepSpeech model
        jprint('Initializing model...')
        dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'speech')
        model_name = glob.glob(os.path.join(dirname,'*.tflite'))[0]
        logging.info("Model: %s", model_name)
        self.model = deepspeech.Model(model_name)
        try:
            scorer_name = glob.glob(os.path.join(dirname,'*.scorer'))[0]
            logging.info("Language model: %s", scorer_name)
            self.model.enableExternalScorer(scorer_name)
        except Exception as e:
            pass
        jprint("Initalized.")

    def transcribe(self):
        jprint("Listening...")
        speak("Hello")
        # Start audio with VAD
        vad_audio = VADAudio(aggressiveness=1, device=None, input_rate=16000, file=None)
        frames = vad_audio.vad_collector()

        # Stream from microphone to DeepSpeech using VAD
        stream_context = self.model.createStream()
        test = 0
        for frame in frames:
            if frame is not None:
                logging.debug("streaming frame")
                stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            else:
                logging.debug("end utterence")
                text = stream_context.finishStream()
                jprint(f"Recognized: {text}")
                vad_audio.destroy()

                # Respond
                speak(getResponse(text))
                return 1
                  

    def run(self):
        keywords = list()
        for x in self._keyword_paths:
            keywords.append(os.path.basename(x).replace('.ppn', '').replace('_compressed', '').split('_')[0])

        print('listening for:')
        for keyword, sensitivity in zip(keywords, self._sensitivities):
            print('- %s (sensitivity: %f)' % (keyword, sensitivity))

        porcupine = None
        pa = None
        audio_stream = None
        try:
            porcupine = Porcupine(
                library_path=self._library_path,
                model_path=self._model_path,
                keyword_paths=self._keyword_paths,
                sensitivities=self._sensitivities)

            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length,
                input_device_index=self._input_device_index)

            while True:
                pcm = audio_stream.read(porcupine.frame_length)

                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                if self._output_path is not None:
                    self._recorded_frames.append(pcm)

                result = porcupine.process(pcm)

                if result >= 0:
                    print('[%s] Detected %s' % (str(datetime.now()), keywords[result]))
                    audio_stream.close()
                    if self.transcribe():
                        audio_stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=porcupine.frame_length, input_device_index=self._input_device_index)

        except KeyboardInterrupt:
            print('stopping ...')
        finally:
            if porcupine is not None:
                porcupine.delete()

            if audio_stream is not None:
                audio_stream.close()

            if pa is not None:
                pa.terminate()

            if self._output_path is not None and len(self._recorded_frames) > 0:
                recorded_audio = np.concatenate(self._recorded_frames, axis=0).astype(np.int16)
                soundfile.write(self._output_path, recorded_audio, samplerate=porcupine.sample_rate, subtype='PCM_16')

    _AUDIO_DEVICE_INFO_KEYS = ['index', 'name', 'defaultSampleRate', 'maxInputChannels']

    @classmethod
    def show_audio_devices_info(cls):
        """ Provides information regarding different audio devices available. """

        pa = pyaudio.PyAudio()

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            print(', '.join("'%s': '%s'" % (k, str(info[k])) for k in cls._AUDIO_DEVICE_INFO_KEYS))

        pa.terminate()

if __name__ == '__main__':
    PorcupineDemo(library_path=LIBRARY_PATH, model_path=MODEL_PATH, keyword_paths=[KEYWORD_PATHS["jarvis"]], sensitivities=[0.9], output_path=None, input_device_index=None).run()