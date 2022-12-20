from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from visionanalytic.detector import FaceDetector
from visionanalytic.recognition import Embedder
from visionanalytic.data import VisionCRM
from visionanalytic.utils import crop_save, insert_face
from config.config import RESULTS_DIR, DATA_DIR

cap = cv2.VideoCapture(str(DATA_DIR/"IMG_0791.MOV"))
#cap = cv2.VideoCapture(0)
rotate = True
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

center = (W / 2, H / 2)
angle = 180
scale = 1

M = cv2.getRotationMatrix2D(center, angle, scale)


skip_fps = 5
n_photo_frames = 2
totalFrame = 0
n_detection = 0
img_sample = None
show_n = 30 * 5
currently_show = 0

face_detection = FaceDetector()
embeder_model = Embedder()
vision_crm = VisionCRM()

for file in Path(DATA_DIR / "all_consumer").iterdir():

    name_file = file.stem
    frame  = cv2.imread(str(file))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    xyxy = face_detection.detect(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if xyxy:
        n_detection += 1
        crop_save(frame, xyxy, Path(RESULTS_DIR , '0000', str(name_file) + ".jpg"))
