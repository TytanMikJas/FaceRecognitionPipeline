from threading import Thread
import time
import cv2
from AppState import AppState
from utils.phase_1 import PhaseOne

STAGE_DELAY = 1.0

class PipelineThread(Thread):
  def __init__(self, app_state: AppState) -> None:
    super().__init__()
    self.app_state = app_state
    self.phase_1 = PhaseOne()
    # self.phase_2 = 
    # self.phase_3 = 

  def run(self) -> None:
    time.sleep(1)
    while self.app_state.pipeline_flag:
      self.run_pipeline()
                        
  def run_pipeline(self)-> None:
      self.app_state.msgs.clear()
      recent_img = self.app_state.photo

      if recent_img is None:
        time.sleep(0.1)
        return 

      msg, img = self.phase_1.run(recent_img.copy()) 
      if msg:
        return 

      self.app_state.camera_lock.acquire()
      self.app_state.use_camera = False
      self.app_state.camera_lock.release()

      self.app_state.photo = cv2.resize(img, (self.app_state.WIDTH, self.app_state.HEIGHT))
      self.app_state.msgs.append("Face detected")
      
      # TODO: run phase 2 here 
      time.sleep(STAGE_DELAY)
      self.app_state.use_camera = True
      

