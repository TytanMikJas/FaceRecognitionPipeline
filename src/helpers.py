import cv2
from constants import A, B, C, FONT_SIZE
from tkinter.font import Font

def get_camera_resolution() -> tuple[int, int]:
  camera = cv2.VideoCapture(0)
  if not camera.isOpened():
    raise Exception("Could not open video device")
  width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
  camera.release()
  return width, height

def compute_formula(hand, low_qual: float, normal: float) -> float:
  return normal*A - (hand*B + low_qual*C)


def find_blame(hand: float, low_qual: float) -> str:
  if hand > low_qual:
    return "Your face looks like its covered"
  return "The image quality or light conditions are poor."


class Msg:
  def __init__(self, text, color='red'):
    self.text = text
    self.color = color