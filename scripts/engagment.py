import cv2
from datetime import datetime
import torch
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from visionanalytic.utils import get_angle, xyxy_to_xywh
from visionanalytic.tracking import TrackableObject, Tracker
from config.config import DATARESEARCH_DIR

def engagement_detect(left_angle, right_angle) -> bool:
    min_angle = 80
    max_angle = 110

    if ((left_angle > min_angle) & (right_angle > min_angle) 
    & (left_angle < max_angle) & (right_angle < max_angle)):
        return True
    else:
        return False

app = FaceAnalysis(
    allowed_modules = ["detection"],
    providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

device = 0
video_path = DATARESEARCH_DIR / "2023-01-05-231506.webm"
# cap = cv2.VideoCapture(str(video_path))
cap = cv2.VideoCapture(0)

skip_frame = 20
total_frame = 0

#frame_front = np.zeros(((840, 640, 3)), dtype=np.uint8)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(str(DATARESEARCH_DIR / "result_engagement.mp4"), fourcc, 30.0, (W, H), True)

pxs_x = 200
pxs_y = 300

x_start = int(W / 2) - int(pxs_x/2)
x_end = int(W / 2) + int(pxs_x/2)
y_start = int(H / 2) - int(pxs_y/2)
y_end = int(H / 2) + int(pxs_y/2)

xyxy_crop = np.array([x_start, y_start, x_end, y_end])
time_count = 0

tracker = Tracker()
last_detection = {}
trackableObjects = {}
time_engagement = {}

while True:

    ret, frame = cap.read()

    # crop frame
    #frame = crop_img(frame, xyxy_crop)

    if not ret:
        break

    if total_frame % skip_frame == 0:
        faces = app.get(frame)

    if not faces:
        rimg = frame
    
    else:
        start_time = datetime.now()

        xyxy = []
        confidence = []
        clss = [0] * len(faces)

        for face in faces:
            xyxy.append(face["bbox"])
            confidence.append(face["det_score"])

        xyxy = torch.tensor(xyxy)

        last_detection["xywh"] = xyxy_to_xywh(xyxy)
        last_detection["confidences"] = torch.tensor(confidence)
        last_detection["clss"] = torch.tensor(clss)

        # get id and centroids dict
        objects = tracker.update(
            last_detection["xywh"],
            last_detection["confidences"],
            last_detection["clss"],
            frame)

        # be sure n object detected is iqual to m object tracked
        if len(objects) == len(faces):

            # loop for each object tracked and add info
            for (n_object, object_item) in enumerate(objects.items()):
                objectID, centroid = object_item

                to = trackableObjects.get(objectID, None)

                trackableObjects[objectID] = to

                #timer
                time_count = time_engagement.get(objectID, None)
                if not time_count:
                    time_count = 0.0
                    time_engagement[objectID] = time_count

                # Draw centroide and ID
                # text = "ID {}".format(objectID)
                # cv2.putText(frame, text, (centroid[0]-5, centroid[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                # cv2.circle(frame, (centroid[0], centroid[1]), 4, (255,0,0), -1)

                #faces[n_object]["object_id"] = objectID
                face = faces[n_object]
                left_angle = get_angle(
                    face["kps"][0],
                    face["kps"][2],
                    face["kps"][3],
                )
                right_angle = get_angle(
                    face["kps"][1],
                    face["kps"][2],
                    face["kps"][4]
                )
                #angles = (left_angle, right_angle)

                # # draw angles
                # cv2.putText(
                #     frame, f"{int(left_angle)}", (int(face["kps"][2][0]) - 25, int(face["kps"][2][1])),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 58), 2
                #     )

                # cv2.putText(
                #     frame, f"{int(right_angle)}", (int(face["kps"][2][0]) + 20, int(face["kps"][2][1])),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 58), 2
                #     )

                is_engagement = engagement_detect(left_angle, right_angle)

                if is_engagement:
                    
                    scale = (face["bbox"][2] - face["bbox"][0]) / 200
                    diff = (datetime.now() - start_time).total_seconds()
                    add_time = time_engagement[objectID] + 0.0333
                    time_engagement[objectID] = add_time# diff 

                    cv2.rectangle(
                    frame,
                    (int(face["bbox"][0]),int(face["bbox"][1])-int(scale*45)), (int(face["bbox"][2]),int(face["bbox"][1])+int(scale*15)),
                    (10, 255, 58),-1)

                    cv2.putText(
                    frame, "Enganchado", (int(face["bbox"][0]), int(face["bbox"][1])-int(scale*20)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2
                    )

                    cv2.putText(
                            frame, f"{time_engagement[objectID]:.3}", (int(face["bbox"][0]), int(face["bbox"][1])+int(scale*12)),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2
                            )

        #rimg = app.draw_on(frame, faces)
        rimg = frame

    cv2.imshow("people detector", rimg)
    writer.write(rimg)

    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()