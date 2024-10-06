# Face Detection

## BlazeFace
![BlazeFace](../assets/BlazeFace.png)
- Google uses it as a face detector in [MediaPipe Studio](https://mediapipe-studio.webapps.google.com/studio/demo/face_detector)
- [Paper](./papers/BlazeFace:%20Sub-millisecond%20Neural%20Face%20Detection%20on%20Mobile%20GPUs.pdf)
- Characteristics:
  - Based on **SSD architecture** - predefined anchor boxes, but less than in SSD (Due to limited variance in human computing smaller feature maps is redundant)
  <img alt="ssd vs blazeface" src="../assets/BlazeFace-anchors.png" width="600px"/>
-  **depthwise convolutions** with kernels 5x5 - decreasing the total amount of bottlenecks required to reach a particular receptive field size, thus reducing the number of parameters and computations
- input image size: 128x128
- outputs (17 values):
  - bounding box: `ymin`, `xmin`, `ymax`, `xmax` (all normalized to [0, 1])
  - facial landmarks: `right_eye_x`, `right_eye_y`, `left_eye_x`, `left_eye_y`, `nose_x`, `nose_y`, `mouth_x`, `mouth_y`, `right_ear_x`, `right_ear_y`, `left_ear_x`, `left_ear_y`

## YOLO
