import torch
import numpy as np
import cv2
import os
import platform
from ultralytics import YOLO

class YoloV8ImageObjectDetection:
    # CUSTOM_MODEL_PATH = r"..\weights\best.pt" 
    current_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path_components = ['..', '..', 'weights', 'best.pt']
    CUSTOM_MODEL_PATH = os.path.join(current_directory, *relative_path_components)

    def __init__(self, chunked: bytes = None, threshold: float = 0.6):

        self._bytes = chunked
        self.threshold = threshold
        self.model = self._load_model()
        self.device = self._get_device()
        self.classes = self.model.names

    def _get_device(self):
        
        if platform.system().lower() == "darwin":
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self):
        model = YOLO(YoloV8ImageObjectDetection.CUSTOM_MODEL_PATH)
        return model

    async def __call__(self):

        frame = self._get_image_from_chunked()
        results = self.model(frame)
        frame, labels = self.plot_boxes(results, frame)
        return frame, labels
    
    def _get_image_from_chunked(self):
        arr = np.asarray(bytearray(self._bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)  # 'Load it as it is'
        return img
    
    def class_to_label(self, x):

        return self.classes[int(x)]

    def plot_boxes(self, results, frame):

        labels_confidences = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf >= self.threshold:
                    c = box.cls
                    l = self.model.names[int(c)]
                    conf = round(box.conf[0].item(), 2)
                    labels_confidences.append((l, conf))
        frame = results[0].plot()
        return frame, labels_confidences
