import pvporcupine
handle = pvporcupine.create(keywords=['jarvis'])

handle.delete()
def get_next_audio_frame():
  pass

try: 
    while True:
        keyword_index = handle.process(get_next_audio_frame())
        if keyword_index >= 0:
            print("hello")
            pass
except KeyboardInterrupt:
    handle.delete()
    print("Press Ctrl-C to terminate while statement")
    pass