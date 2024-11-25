import tkinter as tk
from PIL import Image, ImageTk
import sys
from cv2.typing import MatLike
from typing import Literal
import cv2

sys.path.append('./../')
from utils.phase_1 import take_photo_from_camera, reverse_channels, PhaseOne

class CameraView(tk.Canvas):
  def __init__(self, master, **kwargs):
    super().__init__(master, **kwargs)
    self.camera = cv2.VideoCapture(0)
    self.photo_gen = take_photo_from_camera(self.camera)
    self.img = None
    self.phase_1 = PhaseOne()
    self.coursor_y_pos = 0

  def handle_phases(self, photo: MatLike):
    msg, img = self.phase_1.run(photo) 
    if msg:
      print(msg)
      return
    self.add_text("Face detected", fill='green')
    # TODO: run phase 2 here 


  def add_text(self, text, fill='red', font=("Arial", 20), anchor: Literal['nw']=tk.NW):
    self.coursor_y_pos += 25
    self.create_text(10, self.coursor_y_pos, text=text, fill=fill, font=font, anchor=anchor)

  def update_photo(self):
    photo =  reverse_channels(next(self.photo_gen))
    self.img = ImageTk.PhotoImage(image=Image.fromarray(photo))
    self.create_image(0, 0, image=self.img, anchor=tk.NW)
    self.add_text("Searching for faces...")
    return photo

  def pack(self, **kwargs):
    self.coursor_y_pos = 0
    photo = self.update_photo()
    super().pack(**kwargs)
    self.handle_phases(photo)
    self.after(1, self.pack)

  def destroy(self):
    self.camera.release()
    super().destroy()