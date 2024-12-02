import tkinter as tk

from CameraView import CameraView
from helpers import get_camera_resolution

root = tk.Tk()
root.title("Face Recognition Pipeline")

width, height = get_camera_resolution()

root.geometry(f"{width}x{height}")
root.resizable(False, False)

camera_view = CameraView(root, width=width, height=height)
camera_view.pack()

root.mainloop()
