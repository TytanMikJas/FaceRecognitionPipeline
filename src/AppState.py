from threading import Lock
from cv2.typing import MatLike
from typing import List

class AppState:
  def __init__(self, WIDTH, HEIGHT) -> None:
    self.WIDTH = WIDTH
    self.HEIGHT = HEIGHT

    self.pipeline_flag = True
    self.use_camera = True
    self.camera_lock = Lock()

    self.msgs: List[str] = []
    self.photo: None|MatLike = None
