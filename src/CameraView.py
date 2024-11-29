import tkinter as tk
from PIL import Image, ImageTk
import sys
from typing import Literal
import cv2

sys.path.append('./../')
from PipelineThread import PipelineThread
from utils.phase_1 import take_photo_from_camera, reverse_channels
from AppState import AppState

class CameraView(tk.Canvas):
  def __init__(self, master, **kwargs):
    super().__init__(master, **kwargs)
    self.camera = cv2.VideoCapture(0)
    self.photo_gen = take_photo_from_camera(self.camera)
    
    self.img_tk: None|ImageTk.PhotoImage = None
    self.app_state = AppState(kwargs['width'], kwargs['height'])

    self.pipeline_thread = PipelineThread(self.app_state) 
    self.pipeline_thread.start()

    self.coursor_y_pos = 0

  def add_text(self, text, fill='red', font=("Arial", 20), anchor: Literal['nw']=tk.NW):
    self.coursor_y_pos += 25
    self.create_text(10, self.coursor_y_pos, text=text, fill=fill, font=font, anchor=anchor)

  def update_photo_with_camera(self) -> None:
    self.app_state.photo = reverse_channels(next(self.photo_gen))

  def draw_msgs(self) -> None:
    self.coursor_y_pos = 0
    self.add_text("Searching for faces...")
    for msgs in self.app_state.msgs.copy():
      self.add_text(msgs)

  def draw_photo(self) -> None:
    self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.app_state.photo.copy())) # type: ignore
    self.create_image(0, 0, image=self.img_tk, anchor=tk.NW)

  def pack(self, **kwargs):
    self.app_state.camera_lock.acquire()
    if self.app_state.use_camera:
      self.update_photo_with_camera()
    self.app_state.camera_lock.release()
    
    self.draw_photo()
    self.draw_msgs()

    super().pack(**kwargs)
    self.after(1, self.pack)

  def destroy(self):
    self.camera.release()
    self.app_state.pipeline_flag = False
    self.pipeline_thread.join()
    super().destroy()