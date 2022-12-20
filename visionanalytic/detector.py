from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from visionanalytic.utils import unit_xywh2xyxy

class FaceDetector:
    def __init__(self, model_selection: int=1, min_detection_confidence: float=0.8):
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence

        self.load_model()

    def load_model(self):
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=self.model_selection, min_detection_confidence=self.min_detection_confidence 
            )

    def detect(self, frame, framesize=None):

        if not framesize:
            W = frame.shape[1]
            H = frame.shape[0]
        else:
            W, H = framesize

        results = self.face_detection.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                
                relative_bounding_box = detection.location_data.relative_bounding_box
                xywh = [
                    int(relative_bounding_box.xmin*W), int(relative_bounding_box.ymin*H),
                    int(relative_bounding_box.width*W), int(relative_bounding_box.height*H)]
                xyxy = unit_xywh2xyxy(xywh)
        
        else:
            return None

        return xyxy