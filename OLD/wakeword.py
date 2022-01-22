import struct
import pyaudio
import pvporcupine
import speech_recognition as sr
import pyttsx3
import datetime
import random
from time import ctime
import requests
import json

porcupine = None
pa = None
audio_stream = None

print('[J.A.R.V.I.S] Loading...')

engine=pyttsx3.init('espeak')
voices=engine.getProperty('voices')
engine.setProperty('voice', 'voices[0].id')

def speak(text):
    engine.say(text)
    engine.runAndWait()

def myCommand():
    r = sr.Recognizer()                                                                                   
    with sr.Microphone() as source:                                                                       
        print("[J.A.R.V.I.S] Listening")
        speak("Listening")
        r.pause_threshold =  1
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio, language='en-in')
        print('User: ' + query + '\n')
        
    except sr.UnknownValueError:
        return "nocluewhatyousaid"
    return query.lower()

def exist(query, terms):
    for term in terms:
        if term in query:
            return True
    return False

def respond(query):
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
    elif exist(query, ['are you up']):
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
        time = "The time is %s %s" % (hours, minutes)
        return time
    elif exist(query, ['what is your name', 'tell me your name']):
        return "My name is Jarvis"
    elif exist(query,["ip address", "what is the ip address", "network ip", "what's the IP address"]):
        data = requests.get('https://api.myip.com/')
        data = data.json()
        return "My ip address is %s and located in the %s" % (str(data["ip"]), str(data["country"]))
    else:
        return "I do not understand that question."
    print('[J.A.R.V.I.S] Finished')

try:
    porcupine = pvporcupine.create(keywords=["jarvis"])
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
                    rate=porcupine.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=porcupine.frame_length)

    print('[J.A.R.V.I.S] Loaded')
    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print("[J.A.R.V.I.S] Detected wakeword")
            query = myCommand()
            response = respond(query)
            speak(response)
finally:
    if porcupine is not None:
        porcupine.delete()

    if audio_stream is not None:
        audio_stream.close()

    if pa is not None:
            pa.terminate()