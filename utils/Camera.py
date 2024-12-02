import time
import cv2 
from threading import Thread

import numpy as np

from utils.phase_1 import take_photo_from_camera

class Camera(Thread):
  def __init__(self, camera_id: int|str, patience: int=1):
    super().__init__()
    self.camera = cv2.VideoCapture(camera_id)
    self.last_frame = None
    self.running = False
    self.patience = patience

  def get_last_frame(self) -> np.ndarray:
    while self.last_frame is None:
      time.sleep(0.2)
    frame = self.last_frame
    self.last_frame = None
    return frame
  
  def run(self):
    self.running = True
    for frame in take_photo_from_camera(self.camera, self.patience):
      self.last_frame = frame
      if not self.running:
        break

  def join(self):
    self.running = False
    super().join()
    self.camera.release()