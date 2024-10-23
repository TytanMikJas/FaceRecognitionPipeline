import cv2
import numpy as np

def take_photo() -> cv2.typing.MatLike: # type: ignore
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
  
def downscale_img(img, target_size: tuple[int, int]): ...

def pick_biggest(face_boxes: list[np.ndarray]): ...

def crop_face_from_img(img: np.ndarray, face_box: np.ndarray) -> np.ndarray:
  x, y, w, h = face_box
  return img[y:y+h, x:x+w]