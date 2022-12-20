from typing import List, Dict
import numpy as np
import cv2
import torch

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def unit_xywh2xyxy(box):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    box_ = [box[0] + box[2], box[1] + box[3]]
    return box[:2] + box_


def crop_save(frame, xyxy, path):
    
    xyxy = xyxy.astype(int)
    for i, value in enumerate(xyxy):
        if value < 0:
            xyxy[i] = 0

    croped = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    cv2.imwrite(str(path), croped)


def resize_img(img, width, height):
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)


def insert_face(frame, cropped_face, texts:Dict):

    width = 150
    height = 150

    info_client = texts["info"]

    cropped_face = resize_img(cropped_face, width, height)
    cv2.putText(frame, texts["id_client"], (15, 185), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 10, 58), 2)
    cv2.putText(frame, " se encuentra en la tienda", (10, 205), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 10, 58), 2)

    cv2.putText(frame, "Tipo Cliente: ", (10, 245), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 10, 58), 2)
    cv2.putText(frame, info_client["type_client"], (50, 270), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (58, 10, 255), 2)

    cv2.putText(frame, "Recomendar: ", (10, 290), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 10, 58), 2)
    cv2.putText(frame, info_client["recomendation"], (50, 315), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (58, 10, 255), 2)
    
    cv2.putText(frame, "Descuento: ", (10, 355), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 10, 58), 2)
    cv2.putText(frame, info_client["discount"], (50, 380), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (58, 10, 255), 2)

    cv2.putText(frame, "Ultima visita: ", (10, 410), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 10, 58), 2)
    cv2.putText(frame, info_client["last_visit"], (50, 435), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (58, 10, 255), 2)

    frame[5:5+height, 5:5+width] = cropped_face

    return frame

def to_center_objects(outputs):

    outputs_bbox = outputs[:, :4]
    centroid_x = np.mean([outputs_bbox[:, 0], outputs_bbox[:, 2]], axis=0)
    centroid_y = np.mean([outputs_bbox[:, 1], outputs_bbox[:, 3]], axis=0)
    id_objects = outputs[:, 4]

    center_objects = {}

    for id_object, x, y in zip(id_objects, centroid_x, centroid_y):
        center_objects[id_object] = [int(x), int(y)]
    
    return center_objects

def xyxy_to_xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def generate_notification(face, df_info):
    
    img_notification = np.zeros((210, 640, 3)).astype(np.uint8)
    img_notification[:, :, :] = (0,191,255)

    width = 100
    height = 100

    cv2.putText(img_notification, df_info["name"][0], (30, 35), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 10, 58), 2)
    cv2.putText(img_notification, "se encuentra", (25, 175), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 10, 58), 2)
    cv2.putText(img_notification, "en la tienda", (25, 195), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 10, 58), 2)

    cv2.putText(img_notification, "Tipo Cliente: ", (250, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 10, 58), 2)
    cv2.putText(img_notification, df_info["type_client"][0], (380, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (58, 10, 255), 2)

    cv2.putText(img_notification, "Recomendar: ", (250, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 10, 58), 2)
    cv2.putText(img_notification, df_info["recomendation"][0], (380, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (58, 10, 255), 2)

    cv2.putText(img_notification, "Descuento: ", (250, 130), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 10, 58), 2)
    cv2.putText(img_notification, df_info["descount"][0], (380, 130), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (58, 10, 255), 2)

    cv2.putText(img_notification, "Ultima visita: ", (250, 180), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 10, 58), 2)
    cv2.putText(img_notification, df_info["last_visit"][0], (380, 180), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (58, 10, 255), 2)

    cv2.line(img_notification, (0, 205), (640, 205), (0, 0, 0), 2) 

    cropped_face = cv2.resize(face, (width, height))
    img_notification[50:50+height, 50:50+width] = cropped_face
    
    return (img_notification, df_info["object_id"][0])