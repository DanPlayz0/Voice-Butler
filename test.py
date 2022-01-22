import io, picamera, cv2, numpy

stream = io.BytesIO()
with picamera.PiCamera() as camera:
  camera.resolution = (320, 240)
  camera.capture(stream, format='jpeg')