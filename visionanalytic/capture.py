import threading
import time
from pathlib import Path
from typing import Union

import cv2


class StreamCapture:
    """Multi-threaded video capture"""

    # taken from https://github.com/gilbertfrancois/video-capture-async

    def __init__(self, source: Union[int, str, Path], width=1200, height=1000) -> None:

        self.source = source

        if isinstance(self.source, str):
            if not source.startswith("http://"):
                self.source = Path(source)

        if isinstance(self.source, Path):
            if not self.source.exists():
                raise FileNotFoundError(f"{self.source} does not exist.")
            self.source = str(self.source)

        self.stream = cv2.VideoCapture(self.source)
        # self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.FPS = 30  # self.stream.get(cv2.CAP_PROP_FPS)
        self.FPS_S = 1 / self.FPS
        self.FPS_MS = int(1000 / self.FPS)

        self.grabbed, self.frame = self.stream.read()

        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def get(self, propId):
        return self.stream.get(propId)

    def start(self):

        if self.started:
            return

        self.started = True

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        return self

    def update(self):
        while self.started:

            grabbed, frame = self.stream.read()

            with self.read_lock:
                self.grabbbed = grabbed
                self.frame = frame

            time.sleep(self.FPS_S)

    def read(self):
        with self.read_lock:
            frame = self.frame  # .copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()
