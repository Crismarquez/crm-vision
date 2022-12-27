import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from visionanalytic.utils import crop_img

app = FaceAnalysis(
    allowed_modules = ["detection"],
    providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

device = 0
cap = cv2.VideoCapture(device)

skip_frame = 20
total_frame = 0

frame_front = np.zeros(((840, 640, 3)), dtype=np.uint8)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

pxs_x = 200
pxs_y = 300

x_start = int(W / 2) - int(pxs_x/2)
x_end = int(W / 2) + int(pxs_x/2)
y_start = int(H / 2) - int(pxs_y/2)
y_end = int(H / 2) + int(pxs_y/2)

xyxy_crop = np.array([x_start, y_start, x_end, y_end])

while True:

    ret, frame = cap.read()

    # crop frame
    frame = crop_img(frame, xyxy_crop)

    if not ret:
        break

    if total_frame % skip_frame == 0:
        faces = app.get(frame)

    if not faces:
        rimg = frame
    
    else:
        rimg = app.draw_on(frame, faces)

    cv2.imshow("people detector", rimg)

    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

