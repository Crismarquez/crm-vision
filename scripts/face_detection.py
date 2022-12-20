from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from visionanalytic.detector import FaceDetector
from visionanalytic.recognition import Embedder
from visionanalytic.data import VisionCRM
from visionanalytic.utils import crop_save, insert_face
from config.config import RESULTS_DIR, DATA_DIR

video_name = "IMG_0795.mov"
cap = cv2.VideoCapture(str(DATA_DIR/video_name))

#cap = cv2.VideoCapture(0)
rotate = True
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

W = 600
H = 800

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(str(RESULTS_DIR / video_name), fourcc, 30.0, (W, H), True)

center = (W / 2, H / 2)
angle = 180
scale = 1

M = cv2.getRotationMatrix2D(center, angle, scale)


skip_fps = 10
n_photo_frames = 4
totalFrame = 0
n_detection = 0
img_sample = None
show_n = 30 * 5
currently_show = 0

face_detection = FaceDetector()
embeder_model = Embedder(model_name="DeepFace")
vision_crm = VisionCRM()

while True:
    ref, frame = cap.read()

    if not ref:
        break

    frame = cv2.resize(frame, (W, H))

    if rotate:
        frame = cv2.warpAffine(frame, M, (W, H))

    currently_show += 1
    if totalFrame % skip_fps == 0:
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        xyxy = face_detection.detect(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if xyxy:
            n_detection += 1
            crop_save(frame, xyxy, Path(RESULTS_DIR , '0000', str(n_detection) + ".jpg"))

            #mp_drawing.draw_detection(frame, detection)
            cv2.rectangle(frame, xyxy[:2], xyxy[2:], (255, 0, 0), thickness=2)

        # how many frames
        n_data = len(list(Path(RESULTS_DIR, "0000").iterdir()))
        if n_data > n_photo_frames:
            currently_show  = 0
            print("*** calculating embeding ***")
            embedding = embeder_model.represent("0000")
            
            # search client
            result = vision_crm.search_embbeding(embedding, threshold=60)
            img_sample = cv2.imread(
                str(list(Path(RESULTS_DIR, "0000").iterdir())[int(n_photo_frames/2)])
                )
            client = result['id_client']

            # clean directory
            [photo_path.unlink() for photo_path in Path(RESULTS_DIR, "0000").iterdir()]


            print(f"{result['id_client']} is inside the store, score: {result['similarity']}")

    totalFrame += 1 

    if currently_show < show_n:
        if isinstance(img_sample, np.ndarray):
            frame = insert_face(frame, img_sample, result)

    cv2.imshow("Mediapipe face detection", frame)

    writer.write(frame)

    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()