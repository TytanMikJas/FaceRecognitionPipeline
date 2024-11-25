import cv2

def get_camera_resolution() -> tuple[int, int]:
  camera = cv2.VideoCapture(0)
  if not camera.isOpened():
    raise Exception("Could not open video device")
  width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
  camera.release()
  return width, height
