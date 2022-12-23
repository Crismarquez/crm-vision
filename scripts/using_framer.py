import cv2

from visionanalytic.framer import FeatureExtractorResearch, Framer
from visionanalytic.recognition import FaceRecognition
from visionanalytic.tracking import Tracker
from visionanalytic.data import VisionCRM

from config.config import DATA_DIR


video_path = DATA_DIR / "sample2.MOV"
# video_path = DATA_DIR / "fix-IMG_0783.mov"
source = 0
recognition = FaceRecognition()
tracker = Tracker()

framer = Framer(
    source = video_path,
    recognition=recognition,
    tracker=tracker,
    crm_ddbb=VisionCRM(n_embeddings=15),
    frame_skipping=2,
    write=True
)

framer.capture()

# framer = FeatureExtractor(
#     source = video_path,
#     recognition=recognition,
#     tracker=tracker,
#     frame_skipping=2,
#     write=False,
#     write_raw=False,
#     name_experiment="exp_002_sample1"
# )

# framer.capture()
