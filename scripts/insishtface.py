import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

device = 0
cap = cv2.VideoCapture(device)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)
    rimg = app.draw_on(frame, faces)

    cv2.imshow("people detector", rimg)

    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

