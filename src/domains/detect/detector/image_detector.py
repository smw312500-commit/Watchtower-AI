import os
from abc import abstractmethod, ABC
from pathlib import Path

from ultralytics import YOLO


model_path = Path(os.environ.get("MODELS_FOLDER", "models"))


class AbstractDetector(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, src, **kwargs):
        pass


class DetectorYOLO(AbstractDetector):
    def __init__(self):
        self.model = YOLO(model_path / "yolo11n.pt", verbose=False)

    def detect(self, src, **kwargs):
        return self.model(src, **kwargs)


class DetectorFireDetectV1(AbstractDetector):
    def __init__(self):
        self.model = YOLO(model_path / "fire_detect_v251205_1.pt", verbose=False)

    def detect(self, src, **kwargs):
        return self.model(src, **kwargs)
