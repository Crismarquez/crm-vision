import json
import time
from pathlib import Path
from typing import List, Dict
import datetime

import cv2
import numpy as np
from imutils.video import FPS
import torch
import pandas as pd

from config.config import RESULTS_DIR, IMGCROP_DIR, RAWDATA_DIR
from visionanalytic.recognition import Embedder, FaceRecognition
from visionanalytic.data import VisionCRM, NotificationManager
from visionanalytic.capture import StreamCapture
from visionanalytic.utils import crop_save, xyxy_to_xywh, crop_img
from visionanalytic.tracking import TrackableObject, Tracker


class Framer:
    def __init__(
        self,
        source: str,
        recognition: FaceRecognition,
        tracker: Tracker,
        crm_ddbb: VisionCRM,
        frame_skipping=10,
        write: bool = True
    ) -> None:

        self.frame_skipping = frame_skipping
        self.recognition = recognition
        self.tracker = tracker
        self.crm_ddbb = crm_ddbb

        self.writer_params = {}
        self.write = write

        self.df_stream = pd.DataFrame()
        time_register = time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())

        self.n_embeddings = self.crm_ddbb.n_embeddings

        self.notification_manager = NotificationManager()
        self.front_image = np.zeros((840, 640*2, 3)).astype(np.uint8)
        

        if isinstance(source, int):
            self.reader = StreamCapture(source=source)
            output_file = str(RESULTS_DIR / f"{'out_' + time_register}.avi")

        elif isinstance(source, str):
            if source.startswith("http://"):
                self.reader = StreamCapture(source=source)
                output_file = str(RESULTS_DIR / f"{'out_' + time_register}.avi")

        elif Path(source).is_file():
            self.reader = cv2.VideoCapture(str(source))
            output_file = str(RESULTS_DIR / f"{Path(source).stem}.mov")

        else:
            raise ValueError("Framer configuration was not able to initialize")

        if self.write:
            self.writer_params = {
                    "output_file": output_file,
                    "fourcc": cv2.VideoWriter_fourcc(*'MP4V'),
                    "fps": int(self.reader.get(cv2.CAP_PROP_FPS)),
                    "frameSize": (640*2, 840)
                }

            self.writer = cv2.VideoWriter(
                self.writer_params["output_file"],
                self.writer_params["fourcc"],
                self.writer_params["fps"],
                self.writer_params["frameSize"],
                True
            )


    def capture(self) -> None:

        if isinstance(self.reader, StreamCapture):
            self.reader.start()

        total_frames = 0

        start = time.time()


        fps = FPS().start()

        last_detection = {}
        tracker = Tracker()
        trackableObjects = {}
        raw_data = {}

        
        self.front_notification = self.notification_manager.home_notification.astype(np.uint8)

        # while self.reader.started:
        while True:

            grabbed, frame = self.reader.read()

            if not grabbed:
                break


            # reshape
            frame = cv2.resize(frame, (640, 840))

            if frame is None:
                break

            # skip frames - predict or only tracking
            if total_frames % self.frame_skipping == 0:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                date_time = datetime.datetime.now()

                faces = self.recognition.predict(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if len(faces) > 0:

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

                    # Recorremos cada una de las detecciones
                    for (n_object, object_item) in enumerate(objects.items()):
                        objectID, centroid = object_item

                        faces[n_object]["object_id"] = objectID
                        # Revisamos si el objeto ya se ha contado
                        to = trackableObjects.get(objectID, None)

                        trackableObjects[objectID] = to
                        objects_to_save[objectID] = centroid

                        # crop image, save image and reference
                        bbox = faces[n_object]["bbox"]
                        croped_img = crop_img(frame, bbox)

                        # # Dibujamos el centroide y el ID de la detección encontrada
                        # text = "ID {}".format(objectID)
                        # cv2.putText(frame, text, (centroid[0]-5, centroid[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

                        faces[n_object]["img_crop"] = croped_img

                        if objects_to_save:
                            raw_data[str(date_time)] = objects_to_save

                    if len(objects.items()) > 0:
                        # update stream dataframe
                        self.df_stream = pd.concat(
                            [self.df_stream, pd.DataFrame.from_records(faces)]
                            )

                        # select N=15 embedding to generate prediction
                        count_values = self.df_stream["object_id"].value_counts()
                        to_filter_id = list(count_values[count_values>self.n_embeddings].index)
                        df_to_predict = self.df_stream[self.df_stream["object_id"].isin(to_filter_id)]

                        if len(df_to_predict)>0:
                            df_predict = self.crm_ddbb.predict(df_to_predict, 0.78)
                            # print(df_predict)
                            df_clients_identified = df_predict[
                                df_predict["prediction_object_id_"] != "no identified"
                                ]

                            # TODO update crm ddbb
                            df_clients_not_identified = df_predict[
                                df_predict["prediction_object_id_"] == "no identified"
                                ]

                            # clients in store
                            df_costumers_in_store = pd.merge(
                                df_clients_identified,
                                self.crm_ddbb.df_infoclients,
                                how="left",
                                left_on="prediction_object_id_",
                                right_on="object_id")
                            
                            # self.crm_ddbb.df_infoclients[
                            #     self.crm_ddbb.df_infoclients["object_id"].isin(
                            #         list(df_clients_identified["prediction_object_id_"])
                            #     )
                            # ]

                            for stream_id_object, predict_id_object in df_costumers_in_store[["object_id_x", "prediction_object_id_"]].values:
                                face_crop = self.df_stream[self.df_stream["object_id"]==stream_id_object]["img_crop"].sample(1).values[0]
                                self.front_notification = self.notification_manager.generate_notification(
                                    face_crop,
                                    self.crm_ddbb.df_infoclients[self.crm_ddbb.df_infoclients["object_id"]==predict_id_object]
                                    )
                        
                            # # clean stream
                            self.df_stream = self.df_stream.drop(
                                self.df_stream[self.df_stream["object_id"].isin(
                                    list(df_clients_identified["prediction_object_id_"]))].index
                                )
                                
            else:
                if last_detection:
                    objects_to_save = {}

                    # get id and centroids dict
                    objects = tracker.update(
                        last_detection["xywh"],
                        last_detection["confidences"],
                        last_detection["clss"],
                        frame)

                    # Recorremos cada una de las detecciones sin recognition
                    for (n_object, object_item) in enumerate(objects.items()):
                        objectID, centroid = object_item

                        # Revisamos si el objeto ya se ha contado
                        to = trackableObjects.get(objectID, None)

                        trackableObjects[objectID] = to
                        objects_to_save[objectID] = centroid

                        # # Dibujamos el centroide y el ID de la detección encontrada
                        # text = "ID {}".format(objectID)
                        # cv2.putText(frame, text, (centroid[0]-5, centroid[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

                    if objects_to_save:
                        raw_data[str(date_time)] = objects_to_save

            self.front_image[:, :640] = frame
            self.front_image[:, 640:]= self.front_notification

            cv2.imshow("front-app", self.front_image)

            if self.write:
                self.writer.write(self.front_image)

            if cv2.waitKey(10) == ord("q"):
                break

            total_frames += 1

        self.reader.release()
        if self.write:
            self.writer.release()

        cv2.destroyAllWindows()


class FeatureExtractorResearch:
    def __init__(
        self,
        source: str,
        recognition: FaceRecognition,
        tracker: Tracker,
        frame_skipping=10,
        write: bool = True,
        write_raw: bool = True,
        name_experiment: str = ""
    ) -> None:

        self.frame_skipping = frame_skipping
        self.recognition = recognition
        self.tracker = tracker

        self.writer_params = {}
        self.write = write
        self.write_raw = write_raw
        self.name_experiment = name_experiment

        time_register = time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())

        if self.name_experiment:
            name_experiment_file = self.name_experiment
        else:
            name_experiment_file = time_register

        self.pickle_file = Path(RAWDATA_DIR, name_experiment_file + ".npy")
        np.save(self.pickle_file, [], allow_pickle=True)

        if isinstance(source, int):
            self.reader = StreamCapture(source=source)
            output_file = str(RESULTS_DIR / f"{'out_' + time_register}.avi")

        elif isinstance(source, str):
            if source.startswith("http://"):
                self.reader = StreamCapture(source=source)
                output_file = str(RESULTS_DIR / f"{'out_' + time_register}.avi")

        elif Path(source).is_file():
            self.reader = cv2.VideoCapture(str(source))
            output_file = str(RESULTS_DIR / time_register / f"{Path(source).stem}.avi")

        else:
            raise ValueError("Framer configuration was not able to initialize")

        if self.write:
            self.writer_params = {
                    "output_file": output_file,
                    "fourcc": cv2.VideoWriter_fourcc(*"XVID"),
                    "fps": int(self.reader.get(cv2.CAP_PROP_FPS)),
                    "frameSize": (
                        int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    ),
                }

            self.writer = cv2.VideoWriter(
                self.writer_params["output_file"],
                self.writer_params["fourcc"],
                self.writer_params["fps"],
                self.writer_params["frameSize"],
                True
            )

        if self.write_raw:
            self.writer_raw_params = {
                "output_file": str(RESULTS_DIR / f"{'raw_' + time_register}.avi"),
                "fourcc": cv2.VideoWriter_fourcc(*"XVID"),
                "fps": int(self.reader.get(cv2.CAP_PROP_FPS)),
                "frameSize": (
                    int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            }

            self.writer_raw = cv2.VideoWriter(
                self.writer_raw_params["output_file"],
                self.writer_raw_params["fourcc"],
                self.writer_raw_params["fps"],
                self.writer_raw_params["frameSize"],
                True
            )

    def capture(self) -> None:

        if isinstance(self.reader, StreamCapture):
            self.reader.start()

        total_frames = 0

        start = time.time()


        fps = FPS().start()

        last_detection = {}
        tracker = Tracker()
        trackableObjects = {}
        raw_data = {}


        # while self.reader.started:
        while True:

            grabbed, frame = self.reader.read()

            if not grabbed:
                break

            if frame is None:
                break

            if total_frames % self.frame_skipping == 0:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                date_time = datetime.datetime.now()

                faces = self.recognition.predict(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if len(faces) > 0:

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

                    print('confidence: ', confidence)

                    # get id and centroids dict
                    objects = tracker.update(
                        last_detection["xywh"],
                        last_detection["confidences"],
                        last_detection["clss"],
                        frame)

                    # Recorremos cada una de las detecciones
                    for (n_object, object_item) in enumerate(objects.items()):
                        objectID, centroid = object_item

                        faces[n_object]["object_id"] = objectID
                        # Revisamos si el objeto ya se ha contado
                        to = trackableObjects.get(objectID, None)

                        trackableObjects[objectID] = to
                        objects_to_save[objectID] = centroid

                        # crop image, save image and reference
                        bbox = faces[n_object]["bbox"]
                        img_name = str(objectID) + str(date_time) + ".jpg"
                        img_path = IMGCROP_DIR / img_name
                        crop_save(frame, bbox, img_path)

                        # Dibujamos el centroide y el ID de la detección encontrada
                        text = "ID {}".format(objectID)
                        cv2.putText(frame, text, (centroid[0]-5, centroid[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

                        faces[n_object]["img_crop"] = img_name

                        # update npy
                        all_data = np.load(self.pickle_file, allow_pickle=True)
                        all_data = list(all_data) + faces
                        np.save(self.pickle_file, np.array(all_data), allow_pickle=True)

                        if objects_to_save:
                            raw_data[str(date_time)] = objects_to_save

            else:
                if last_detection:
                    objects_to_save = {}

                    # get id and centroids dict
                    objects = tracker.update(
                        last_detection["xywh"],
                        last_detection["confidences"],
                        last_detection["clss"],
                        frame)

                    # Recorremos cada una de las detecciones sin recognition
                    for (n_object, object_item) in enumerate(objects.items()):
                        objectID, centroid = object_item

                        # Revisamos si el objeto ya se ha contado
                        to = trackableObjects.get(objectID, None)

                        trackableObjects[objectID] = to
                        objects_to_save[objectID] = centroid

                        # Dibujamos el centroide y el ID de la detección encontrada
                        text = "ID {}".format(objectID)
                        cv2.putText(frame, text, (centroid[0]-5, centroid[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

                    if objects_to_save:
                        raw_data[str(date_time)] = objects_to_save

            cv2.imshow("face detector", frame)

            if cv2.waitKey(10) == ord("q"):
                break

            total_frames += 1

        self.reader.release()

        cv2.destroyAllWindows()



            

            

2