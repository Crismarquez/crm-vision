from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
ASSETS_DIR = Path(DATA_DIR, "assets")
STORE_DIR = Path(BASE_DIR, "store")
MODELS_DIR = Path(STORE_DIR, "models")
RESULTS_DIR = Path(BASE_DIR, "results")
IMGCROP_DIR = Path(RESULTS_DIR, "img_crop")
RAWDATA_DIR = Path(RESULTS_DIR, "raw_data")

# Add to path
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "strong_sort"))

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMGCROP_DIR.mkdir(parents=True, exist_ok=True)
RAWDATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

BACKENDS = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]