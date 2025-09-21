# src/services/scene_detector_service.py

import logging
from pathlib import Path
from typing import Dict, List

# This service uses the PySceneDetect library for robust scene detection.
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

logger = logging.getLogger(__name__)


class SceneDetectorService:
    """
    Detects scene boundaries in a video file using the PySceneDetect library.

    This service acts as the first stage in the reframing pipeline, splitting the
    source video into a list of continuous scenes. This ensures that subject
    tracking is logically constrained to a single shot.
    """

    def __init__(self, threshold: float = 27.0):
        """
        Initializes the scene detector with a configurable threshold.

        Args:
            threshold: The sensitivity for the ContentDetector. Lower values
                       result in more scenes being detected.
        """
        self.threshold = threshold
        logger.info(f"SceneDetectorService initialized with threshold: {self.threshold}.")

    def run(self, media_path: Path) -> List[Dict[str, int]]:
        """
        Analyzes a video to find all continuous scenes.

        Args:
            media_path: The path to the local video file.

        Returns:
            A list of dictionaries, where each dictionary represents a scene
            with its 'start_frame' and 'end_frame' numbers.
        """
        logger.info(f"Detecting scene boundaries for {media_path.name}...")

        try:
            video = open_video(str(media_path))
            scene_manager = SceneManager()
            
            # Add a ContentDetector to find scene changes based on visual content.
            scene_manager.add_detector(ContentDetector(threshold=self.threshold))
            
            # Perform the scene detection.
            scene_manager.detect_scenes(video=video, show_progress=False)
            
            # Retrieve the list of detected scenes.
            scene_list = scene_manager.get_scene_list()

            # Format the output into the pipeline's internal standard format.
            scenes = []
            if not scene_list:
                # Handle the edge case where no cuts are found; the whole video is one scene.
                total_frames = int(video.frame_rate * video.duration.get_seconds())
                scenes.append({"start_frame": 0, "end_frame": total_frames - 1})
            else:
                for scene in scene_list:
                    start_frame = scene[0].get_frames()
                    # The library's end frame is the start of the next scene, so we
                    # subtract 1 to get the inclusive last frame of the current scene.
                    end_frame = scene[1].get_frames() - 1
                    scenes.append({"start_frame": start_frame, "end_frame": end_frame})

            logger.info(f"Scene detection complete. Found {len(scenes)} scenes.")
            return scenes

        except Exception as e:
            logger.error(f"PySceneDetect failed to process the video: {e}", exc_info=True)
            return []
