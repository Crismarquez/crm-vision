from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from deepface import DeepFace
import insightface
from insightface.app import FaceAnalysis

from config.config import RESULTS_DIR

class FaceRecognition:
    def __init__(
        self,
        providers: List = ['CPUExecutionProvider'],
        allowed_modules: List = ["detection", "recognition"],
        det_size: Tuple = (640, 640)
        ) -> None:
        
        self.providers = providers
        self.allowed_modules = allowed_modules
        self.det_size = det_size

        self.load_model()


    def load_model(self):
        self.model = FaceAnalysis(
            providers=self.providers,
            allowed_modules = self.allowed_modules
            )
        self.model.prepare(ctx_id=0, det_size=self.det_size)


    def _process(self, img: np.ndarray):
        """raw function to use insightface prediction on a image

        Args:
            img (np.ndarray): frame to detect faces

        Returns:
            List[Dict]: List with each face detected, and the dict contains, (depend of modules): 
                'bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose',
                'landmark_2d_106', 'gender', 'age', 'embedding'
                """

        faces = self.model.get(img)
        return faces

    def _clean_output(self, faces: List):

        clean_faces = []
        for face in faces:
            clean_faces.append({key: value for key, value in face.items()})

        return clean_faces


    def predict(self, img: np.ndarray, threshold: float = 0.7):
        faces = self._process(img)

        faces_filtered = [face for face in faces if face["det_score"] > threshold]
        faces_filtered = self._clean_output(faces_filtered)

        return faces_filtered


class SequentialRecognition:
    def __init__(self, model="mean") -> None:
        self.model = model
    
    def predict(self, df_input):

        # transform embeddings

        if self.model == "mean":
            df_predict = df_input.groupby("object_id")["embedding"].mean().reset_index()
            return df_predict

        else:
            raise ValueError(f"model instanced ({self.model}) is not allowed")


class Embedder:
    def __init__(self, model_name: str=" ", detector_backend: str = "mediapipe"):
        self.model_name = model_name
        self.detector_backend = detector_backend

        self.load_model()

    def load_model(self):
        self.model = DeepFace.build_model(self.model_name)

    def _represent(self, path_id_consumer: Path):

        X_consumer = []
        for file in path_id_consumer.iterdir():
            embedding = DeepFace.represent(
            img_path = str(file),
            model = self.model,
            detector_backend=self.detector_backend, 
            enforce_detection=False)
            X_consumer.append(embedding)

        embedding_1d = np.array(X_consumer).mean(axis=0)
        return embedding_1d
    
    def represent(self, id_consumer: str):
        path_id_consumer = Path(RESULTS_DIR, id_consumer)
        embedding = self._represent(path_id_consumer)

        return embedding

