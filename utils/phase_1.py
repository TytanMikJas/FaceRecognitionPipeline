import cv2
import numpy as np
from cv2.typing import MatLike
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import torch

import sys
sys.path.append('./../')
from utils.libs.BlazeFace.blazeface import BlazeFace


def take_photo() -> MatLike: # type: ignore
  camera = cv2.VideoCapture(0)
  if not camera.isOpened():
    raise Exception("Could not open video device")
  ok = False
  while not ok:
    ok, frame = camera.read()
    if not ok:
      continue
    
    camera.release()
    return frame
  
def reverse_channels(img: MatLike) -> MatLike:
  return img[:, :, ::-1] # equivalent to cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def downsize_img(img: MatLike, target_size: tuple[int, int]):
  return cv2.resize(img, target_size)

def get_torch_device():
  print("PyTorch version:", torch.__version__)
  print("CUDA version:", torch.version.cuda)
  print("cuDNN version:", torch.backends.cudnn.version())
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  return device

def load_blazeface(path:str, device: torch.device) -> BlazeFace:
  model = BlazeFace().to(device)
  model.load_weights(path + "blazeface.pth")
  model.load_anchors(path + "anchors.npy")
  return model

def configure_params(model, min_score_thresh, min_suppression_threshold):
  model.min_score_thresh = min_score_thresh
  model.min_suppression_threshold = min_suppression_threshold

def plot_detections(img, detections, with_keypoints=True):
    _, ax = plt.subplots(1, figsize=(5, 5))
    ax.grid(False)
    ax.imshow(img)
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        y_min = detections[i, 0] * img.shape[0]
        x_min = detections[i, 1] * img.shape[1]
        y_max = detections[i, 2] * img.shape[0]
        x_max = detections[i, 3] * img.shape[1]

        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=1, edgecolor="g", facecolor="none", 
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        CIRCLE_SETTINGS = {"radius": 0.5, "linewidth": 2, "edgecolor": "r", "facecolor": "none"}
        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = Circle((kp_x, kp_y), **CIRCLE_SETTINGS, alpha=detections[i, 16])
                ax.add_patch(circle)
    plt.show()


def get_idx_of_biggest_face(detections: np.ndarray) -> int: 
   return int(np.apply_along_axis(lambda x: (x[2] - x[0]) * (x[3] - x[1]), 1, detections).argmax())

def reset_face_angle(img: MatLike, detections: list) -> MatLike:
  right_eye = (int(detections[4] * img.shape[1]), int(detections[5] * img.shape[0]))
  left_eye = (int(detections[6] * img.shape[1]), int(detections[7] * img.shape[0]))

  w, h = img.shape[:2]

  angle = np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])
  angle = np.degrees(angle)

  M = cv2.getRotationMatrix2D(right_eye, angle, scale=1)
  rotated = cv2.warpAffine(img, M, (w, h))

  return rotated

class Margin:
  def __init__(self, top: int|str, right: int|str, left: int|str, bottom: int|str):
      self.top = top
      self.right = right
      self.left = left
      self.bottom = bottom

  def get_margin(self, value: int, margin: int|str):
    if isinstance(margin, int):
      return margin
    elif margin.endswith("%"):
      return int(value * int(margin[:-1]) / 100)
  

  def recalculate_coordinates(self, y_min, x_min, y_max, x_max):
     y_min -= self.get_margin(y_min, self.top)
     x_min -= self.get_margin(x_min, self.left)
     y_max += self.get_margin(y_max, self.bottom)
     x_max += self.get_margin(x_max, self.right)

     return y_min, x_min, y_max, x_max  
   

def crop_face_from_img(img: MatLike, detections: MatLike, margin: Margin) -> MatLike:
  y_min = int(detections[0] * img.shape[0])
  x_min = int(detections[1] * img.shape[1])
  y_max = int(detections[2] * img.shape[0])
  x_max = int(detections[3] * img.shape[1])

  y_min, x_min, y_max, x_max = margin.recalculate_coordinates(y_min, x_min, y_max, x_max)
  return img[y_min:y_max, x_min:x_max]

BLAZEFACE_INPUT_SIZE = (128, 128)

class PhaseOne:
  def __init__(self, margin: Margin = Margin("30%", 5, 15, 5)):
    self.device = get_torch_device()
    self.margin = margin 
    self.model = load_blazeface("../utils/libs/BlazeFace/", self.device)
    configure_params(self.model, 0.75, 0.3)

  def _predict(self, img: MatLike):
    downsized_img = downsize_img(img, BLAZEFACE_INPUT_SIZE)
    # view_img(downsized_img)
    detections = self.model.predict_on_image(downsized_img)
    print(detections)
    return detections
  
  def run(self, img: MatLike) -> MatLike | str:
    detections = self._predict(img)

    if len(detections) == 0:
      return "No face detected"
    
    biggest_face_idx = get_idx_of_biggest_face(detections)
    img = reset_face_angle(img, detections[biggest_face_idx])
    
    detections = self._predict(img)
    biggest_face_idx = get_idx_of_biggest_face(detections)
    img = crop_face_from_img(img, detections[biggest_face_idx], self.margin)

    return img


if __name__ == "__main__": 
   # dont run it from project root. Enter any folder like /utils/ or /src/ 
   phaseOne = PhaseOne()
   img = cv2.imread("../assets/the-office-handshake.jpg")
   img = reverse_channels(img)
   out = phaseOne.run(img)
   if isinstance(out, str):
     print(out)
     exit()

   cv2.imshow("Face", cv2.cvtColor(out, cv2.COLOR_BGR2RGB)) 
   cv2.waitKey(0)