
## GENERAL
# You have to have webCam installed on your phone
# use CAMERA=0 if want to use default PC camera
CAMERA: int|str = 'http://192.168.1.141:4747/video'
PATIENCE = 2
STAGE_DELAY = 0.5
FONT_SIZE = 15
DEBUG = True
POSITIVE_COLOR = "green"


## PHASE 2
A, B, C = 1, 0.5, 0.5
THRESHOLD = 0.0
MODEL_PATH ='../models/phase_2/BigModel.keras'

## PHASE 3
DB_PATH = '../database'