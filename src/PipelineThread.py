from threading import Thread
import time
import cv2
from AppState import AppState
from utils.phase_1 import PhaseOne
from utils.phase_2 import PhaseTwo
from constants import DB_PATH, MODEL_PATH, POSITIVE_COLOR, STAGE_DELAY, THRESHOLD
from helpers import Msg, compute_formula, find_blame
from utils.phase_3 import UNKNOWN, PhaseThree
import numpy as np

class PipelineThread(Thread):
  def __init__(self, app_state: AppState) -> None:
    super().__init__()
    self.app_state = app_state
    self.phase_1 = PhaseOne()
    self.phase_2 = PhaseTwo(MODEL_PATH)
    self.phase_3 = PhaseThree(DB_PATH)

  def run(self) -> None:
    time.sleep(.5)
    while self.app_state.pipeline_flag:
      self.run_pipeline()

  def run_phase_1(self, img: np.ndarray) -> None|np.ndarray:
      msg, img = self.phase_1.run(img) 
      if msg:
        return None
      self.app_state.msgs.append(Msg("Face detected"))
      return cv2.resize(img, (self.app_state.WIDTH, self.app_state.HEIGHT))

  def run_phase_2(self, img: np.ndarray) -> bool:
      hand, low_qual, normal = self.phase_2.predict(img)

      print(f"Hand: {hand}, Low quality: {low_qual}, Normal: {normal}") 
      if (value := compute_formula(hand, low_qual, normal)) < THRESHOLD:
        print("Score pip2:", value)
        self.app_state.msgs.append(Msg(find_blame(hand, low_qual))) 
        return False 
      self.app_state.msgs.append(Msg("Quality check passed"))
      return True

  def run_phase3(self, img: np.ndarray):
      dist, identity = self.phase_3.recognize(img)
      if identity == UNKNOWN:
        self.app_state.msgs.append(Msg(f"Could not recognize the person, distance: {dist}"))
      else:
        self.app_state.msgs.append(Msg(f"Hi {identity}! Uncertainty={round(dist)}", POSITIVE_COLOR))

  def run_pipeline(self)-> None:
      self.app_state.msgs.clear()
      self.app_state.msgs.append(Msg("Searching for faces..."))
      recent_img = self.app_state.photo

      if recent_img is None:
        time.sleep(2)
        return 

      cropped_face = self.run_phase_1(recent_img)
      time.sleep(STAGE_DELAY)
      if cropped_face is None:
        return
      self.app_state.camera_lock.acquire()
      self.app_state.use_camera = False
      self.app_state.camera_lock.release()

      self.app_state.photo = cropped_face 
      
      if not (proceed := self.run_phase_2(cropped_face)):
        time.sleep(STAGE_DELAY * 5)
        self.app_state.use_camera = True
        return

      self.run_phase3(cropped_face)

      time.sleep(STAGE_DELAY * 5)
      self.app_state.use_camera = True
      

