
import time
import pickle

import numpy as np

from visionanalytic.framer import FramerCRM
from visionanalytic.recognition import FaceRecognition
from visionanalytic.tracking import Tracker
from visionanalytic.data import CRMProcesor
import config.config


if __name__ == "__main__":
    print("\n")
    print("*"*30)
    print("Welcome to CRM-Vision  \n")
    print("*"*30)
    print("\n")
    print("Setup:  \n")

    source = input("Source camera number: ")
    source = int(source)

    recognition = FaceRecognition(det_size=(320, 320))
    tracker = Tracker()

    framer = FramerCRM(
        source = source,
        recognition=recognition,
        tracker=tracker,
        crm_ddbb=CRMProcesor(id_col="id"),
        frame_skipping=2,
        write=False
    )

    framer.capture()