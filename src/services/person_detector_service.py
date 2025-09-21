# src/services/person_detector_service.py

import logging
from typing import Any, Dict, List

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class PersonDetectorService:
    """
    Detects a person in a single image frame using MediaPipe Pose and calculates
    a bounding box from the resulting pose landmarks.
    """

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        """
        Initializes the MediaPipe Pose model.

        Args:
            model_complexity: Complexity of the pose landmark model: 0, 1, or 2.
            min_detection_confidence: Minimum confidence value (0.0 to 1.0) for the
                                      person detection to be considered successful.
        """
        try:
            self.pose_detector = mp.solutions.pose.Pose(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence
            )
            logger.info("PersonDetectorService initialized successfully with MediaPipe Pose.")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Pose: {e}", exc_info=True)
            self.pose_detector = None

    def run(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processes a single video frame to detect the pose of a person and
        calculates an enclosing bounding box.

        Args:
            frame: The video frame, represented as a NumPy array in BGR format.

        Returns:
            A list containing a single dictionary for the most prominent detected
            person, with a bounding box and a confidence score. Returns an empty
            list if no person is detected with sufficient confidence.
        """
        if self.pose_detector is None:
            logger.error("Pose detector is not initialized; cannot run detection.")
            return []

        frame_height, frame_width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(image_rgb)
        
        detected_persons = []
        if results.pose_landmarks:
            # --- Custom Bounding Box Calculation Logic ---
            # The MediaPipe Pose model provides 33 landmarks but no direct bounding box.
            # This logic algorithmically calculates a tight bounding box around all
            # landmarks that are detected with a high degree of visibility.
            landmarks = results.pose_landmarks.landmark
            
            # 1. Filter for only the landmarks that are confidently detected.
            visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.5]
            if not visible_landmarks:
                return []

            # 2. Find the min/max coordinates among all visible landmarks.
            x_min = min([lm.x for lm in visible_landmarks])
            y_min = min([lm.y for lm in visible_landmarks])
            x_max = max([lm.x for lm in visible_landmarks])
            y_max = max([lm.y for lm in visible_landmarks])
            
            # 3. Use the average landmark visibility as a proxy for confidence.
            confidence = np.mean([lm.visibility for lm in visible_landmarks])

            # 4. Convert the normalized 0-1 coordinates to absolute pixel coordinates.
            abs_box = {
                "x": int(x_min * frame_width),
                "y": int(y_min * frame_height),
                "width": int((x_max - x_min) * frame_width),
                "height": int((y_max - y_min) * frame_height),
            }

            detected_persons.append({
                "box_pixels": abs_box,
                "confidence": confidence
            })

        return detected_persons
