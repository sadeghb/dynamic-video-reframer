# src/services/face_detector_service.py

import logging
from typing import Any, Dict, List

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetectorService:
    """
    Detects faces in a single image frame using the MediaPipe FaceDetection model.
    """

    def __init__(self, model_selection: int = 1, min_detection_confidence: float = 0.5):
        """
        Initializes the MediaPipe FaceDetection model.

        Args:
            model_selection: 0 for short-range models (<2m), 1 for full-range (<5m).
            min_detection_confidence: Minimum confidence value (0.0 to 1.0) for a
                                      detection to be considered successful.
        """
        try:
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence
            )
            logger.info("FaceDetectorService initialized successfully with MediaPipe.")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe FaceDetection: {e}", exc_info=True)
            self.face_detector = None

    def run(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processes a single video frame to detect all present faces.

        Args:
            frame: The video frame, represented as a NumPy array in BGR format.

        Returns:
            A list of dictionaries, where each dictionary represents a detected
            face with its bounding box in absolute pixel coordinates and a
            confidence score.
        """
        if self.face_detector is None:
            logger.error("Face detector is not initialized; cannot run detection.")
            return []

        frame_height, frame_width, _ = frame.shape
        
        # 1. Convert the BGR image (used by OpenCV) to RGB, which MediaPipe expects.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Process the image with MediaPipe to get detection results.
        results = self.face_detector.process(image_rgb)
        
        # 3. Extract and format the results into the pipeline's standard format.
        detected_faces = []
        if results.detections:
            for detection in results.detections:
                # MediaPipe provides a *relative* bounding box. This logic converts
                # it to the absolute pixel coordinates required by the rest of our pipeline.
                relative_box = detection.location_data.relative_bounding_box
                
                abs_box = {
                    "x": int(relative_box.xmin * frame_width),
                    "y": int(relative_box.ymin * frame_height),
                    "width": int(relative_box.width * frame_width),
                    "height": int(relative_box.height * frame_height),
                }

                detected_faces.append({
                    "box_pixels": abs_box,
                    "confidence": detection.score[0]
                })

        return detected_faces
