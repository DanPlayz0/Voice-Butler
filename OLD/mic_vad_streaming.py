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
from time import ctime
import requests

logging.basicConfig(level=logging.INFO)

def jprint(str):
    print(f"[J.A.R.V.I.S] {str}")

jprint('Loading...')

engine=pyttsx3.init('espeak')
voices=engine.getProperty('voices')
engine.setProperty('voice', 'en-n')

def speak(text):
    print(f"JARVIS: {text}")
    engine.say(text)
    engine.runAndWait()

def exist(query, terms):
    for term in terms:
        if term in query:
            return True
    return False

def getResponse(query):
    if exist(query, ["nocluewhatyousaid"]):
        # This part is for when it didnt understand the command.
        return ("Sorry sir! I didn't get that!")
    elif exist(query, ["what do you do", "what are you"]):
        return ("Allow me to introduce myself. I am JARVIS. A virtual artificial intelligence and I'm here to assist you with a variety of tasks the best I can, 24 hours a day, seven days a week.")
    elif exist(query, ['nothing','stop','abort','shut up', 'shut', 'quiet']):
        return ("Bye Sir")
    elif exist(query, ['hello', 'hey']):
        return ("Hello Sir")
    elif exist(query,['bye','goodbye']):
        return ("Bye Sir, have a good day.")
    elif exist(query, ['how are you', "what's up"]):
        stMsgs = ["Just doing my thing!", "I am fine!", "I have no feelings."]
        return (random.choice(stMsgs))
    elif exist(query, ['are you up', 'how are you doing']):
        return ("For you sir, always.")
    elif exist(query, ['execute order']):
        return ("You are not authorized to execute that order.")
    elif exist(query, ['time is it']):
        time = ctime().split(" ")[3].split(":")[0:2]
        if time[0] == "00":
            hours = 12
        else:
            hours = time[0]
        minutes = time[1]
        time = f"The time is {hours} {minutes}"
        return time
    elif exist(query, ['what is your name', 'tell me your name']):
        return "My name is Jarvis"
    else:
        return "I did not understand that."


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
        dirname = os.path.dirname(os.path.abspath(__file__))
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
        # Start audio with VAD
        vad_audio = VADAudio(aggressiveness=1, device=None, input_rate=16000, file=None)
        jprint("Listening...")
        speak("Listening...")
        frames = vad_audio.vad_collector()

        # Stream from microphone to DeepSpeech using VAD
        spinner = Halo(spinner='line')
        stream_context = self.model.createStream()
        #wav_data = bytearray()
        for frame in frames:
            if frame is not None:
                if spinner: spinner.start()
                logging.debug("streaming frame")
                stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
                #if ARGS.savewav: wav_data.extend(frame)
            else:
                if spinner: spinner.stop()
                logging.debug("end utterence")
                #if ARGS.savewav:
                #    vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                #    wav_data = bytearray()
                text = stream_context.finishStream()
                print("Recognized: %s" % text)
                vad_audio.destroy()
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
                                    audio_stream = pa.open(
                                    rate=porcupine.sample_rate,
                                    channels=1,
                                    format=pyaudio.paInt16,
                                    input=True,
                                    frames_per_buffer=porcupine.frame_length,
                                    input_device_index=self._input_device_index)

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
    PorcupineDemo(library_path=LIBRARY_PATH, model_path=MODEL_PATH, keyword_paths=[KEYWORD_PATHS["jarvis"]], sensitivities=[0.5], output_path=None, input_device_index=None).run()