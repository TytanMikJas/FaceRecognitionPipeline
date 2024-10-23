import cv2
import numpy as np

def take_photo():
  camera = cv2.VideoCapture(0)
  if not camera.isOpened():
    raise Exception("Could not open video device")
  ok = False
  while not ok:
    ok, frame = camera.read()
    if not ok:
      continue
    
    # cv2.imwrite("debug-photo.jpg", frame)
    camera.release()
    return frame