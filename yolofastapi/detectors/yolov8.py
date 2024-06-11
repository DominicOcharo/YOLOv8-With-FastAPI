# yolov8.py inside detectors sub directory:
# For machine learning
import torch
# For array computations
import numpy as np
# For image decoding / editing
import cv2
# For environment variables
import os
# For detecting which ML Devices we can use
import platform
# For actually using the YOLO models
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
        """Gets best device for your system

        Returns:
            device (str): The device to use for YOLO for your system
        """
        if platform.system().lower() == "darwin":
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self):
        """Loads a custom-trained YOLO model from a specified path on disk

        Returns:
            model (Model): Trained PyTorch model
        """
        model = YOLO(YoloV8ImageObjectDetection.CUSTOM_MODEL_PATH)
        return model

    async def __call__(self):
        """This function is called when class is executed.
        It analyzes a single image passed to its constructor
        and returns the annotated image and its labels
        
        Returns:
            frame (numpy.ndarray): Frame with bounding boxes and labels plotted on it.
            labels (list(str)): The corresponding labels that were found
        """
        frame = self._get_image_from_chunked()
        results = self.model(frame)
        frame, labels = self.plot_boxes(results, frame)
        return frame, labels
    
    def _get_image_from_chunked(self):
        """Loads an openCV image from the raw image bytes passed by the API.

        Returns: 
            img (numpy.ndarray): opencv2 image object from the raw binary
        """
        arr = np.asarray(bytearray(self._bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)  # 'Load it as it is'
        return img
    
    def class_to_label(self, x):
        """For a given label value, return corresponding string label.
        Arguments:
            x (int): numeric label

        Returns:   
            class (str): corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """Takes a frame and its results as input, 
        and plots the bounding boxes and label on to the frame.

        Arguments:
            results (list(ultralytics.engine.results.Results)): contains labels and coordinates predicted by model on the given frame.
            frame (numpy.ndarray): Frame which has been scored.
        
        Returns:
            frame (numpy.ndarray): Frame with bounding boxes and labels plotted on it.
            labels (list(str)): The corresponding labels that were found
        """
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
    
