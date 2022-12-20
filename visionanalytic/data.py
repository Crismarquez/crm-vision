
from typing import List
import pickle

import pandas as pd
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity


from config.config import DATA_DIR

class VisionCRM:
    def __init__(self, bbdd: str="crm_vision", id_col="object_id", n_embeddings=15) -> None:

        # Load data (deserialize)
        with open(DATA_DIR / f"{bbdd}_auto.pickle", 'rb') as handle:
            unserialized_df = pickle.load(handle)
        self.df_embedding = pd.DataFrame(unserialized_df)

        with open(DATA_DIR / f"{bbdd}_info_client.pickle", 'rb') as handle:
            unserialized_df = pickle.load(handle)
        self.df_infoclients = pd.DataFrame(unserialized_df)

        self.consumers = self.df_embedding[id_col]
        self.n_embeddings = n_embeddings

        self.crm_vision_matrix = [embbeding for embbeding  in self.df_embedding["embedding"].values]

        self.info_client = {
            "type_client": "",
            "recomendation": "",
            "last_visit": "",
            "discount": ""
        }

    def calculate_cosine_similarity(self, matrix_input, matrix_crm):
        return cosine_similarity(
            matrix_input,
            matrix_crm
        )

    def _predict(self, crm_vision_matrix, matrix_input):
    
        #calculate distance
        matrix_similarity = self.calculate_cosine_similarity(
            matrix_input,
            crm_vision_matrix
        )
        
        score = matrix_similarity.max(axis=1)
        arg_max = matrix_similarity.argmax(axis=1)
        
        # get object_id
        predictions = []
        for max_value in arg_max:
            predictions.append(self.consumers.values[max_value])
        
        return predictions, score


    def predict(self, df, distance_treshold=0.8):

        if len(df) == 0:
            return df

        # transform embeddings
        df_predict = df.groupby("object_id")["embedding"].mean().reset_index()

        matrix_input = [embbeding for embbeding  in df_predict["embedding"].values]
        
        # calculate distance
        predictions, score = self._predict(self.crm_vision_matrix, matrix_input)
        
        df_predict["raw_prediction_object_id"] = predictions
        df_predict["score"] = score

        # compare with threshold
        df_predict["prediction_object_id_"] = [
            raw_prediction if score>distance_treshold else "no identified" for raw_prediction, score in df_predict[["raw_prediction_object_id", "score"]].values
        ]

        return df_predict


    def search_embbeding(self, embedding: np.ndarray, threshold=0.5):

        euclidean_distance = np.linalg.norm(self.df_values.values - embedding, axis=1)
        min_pos = np.argmin(euclidean_distance)
        min_distance = euclidean_distance[min_pos]

        if min_distance > threshold:
            return {
                "id_client": "not identified",
                "similarity": min_distance, 
                "info": self.info_client}
        
        id_client = str(self.consumers.values[min_pos][0])

        if id_client in self.consumers.values:
            record = self.df_infoclients[self.df_infoclients["consumer"]==id_client]
            _, type_client, recomendation, last_visit, discount = record.values[0].tolist()
            info_client_result = {
            "type_client": type_client,
            "recomendation": recomendation,
            "last_visit": last_visit,
            "discount": discount
        }

        return {"id_client": id_client, "similarity": min_distance, "info": info_client_result}


class NotificationManager:
    def __init__(self):
        self.showing_ids = []
        self.front_notification = np.ones((840, 640, 3))
        self.front_notification = (self.front_notification*255).astype(np.uint8)
        
        self.height_img_notification = 210
        self.width_img_notification = 640
    
    def _generate_notification(self, face, df_info):
        img_notification = np.zeros(
            (self.height_img_notification, self.width_img_notification, 3)).astype(np.uint8)
        img_notification[:, :, :] = (255,191,0)

        width = 100
        height = 100

        cv2.putText(img_notification, df_info["name"].values[0], (30, 35), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 10, 58), 2)
        cv2.putText(img_notification, "se encuentra", (25, 175), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 10, 58), 2)
        cv2.putText(img_notification, "en la tienda", (25, 195), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 10, 58), 2)

        cv2.putText(img_notification, "Tipo Cliente: ", (250, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 10, 58), 2)
        cv2.putText(img_notification, df_info["type_client"].values[0], (360, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (58, 10, 255), 2)

        cv2.putText(img_notification, "Recomendar: ", (250, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 10, 58), 2)
        cv2.putText(img_notification, df_info["recomendation"].values[0], (360, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (58, 10, 255), 2)

        cv2.putText(img_notification, "Descuento: ", (250, 130), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 10, 58), 2)
        cv2.putText(img_notification, df_info["descount"].values[0], (360, 130), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (58, 10, 255), 2)

        cv2.putText(img_notification, "Ultima visita: ", (250, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 10, 58), 2)
        cv2.putText(img_notification, df_info["last_visit"].values[0], (360, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (58, 10, 255), 2)

        cv2.line(img_notification, (0, 205), (640, 205), (0, 0, 0), 2) 

        cropped_face = cv2.resize(face, (width, height))
        img_notification[50:50+height, 50:50+width] = cropped_face

        return img_notification
    
    def generate_notification(self, face, df_info):
        
        id_object = df_info["object_id"].values[0]
        
        if id_object in self.showing_ids:
            return self.front_notification
        
        img = self._generate_notification(face, df_info)
        
        ocupated_spaces = len(self.showing_ids)
        
        # insert notification
        
        initital_pos = ocupated_spaces * self.height_img_notification
        print(initital_pos)
        self.front_notification[initital_pos:initital_pos+self.height_img_notification] = img
        
        self.showing_ids.append(id_object)
        
        return self.front_notification
    
        