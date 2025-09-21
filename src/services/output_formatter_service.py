# src/services/output_formatter_service.py

import copy
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class OutputFormatterService:
    """
    Transforms the pipeline's internal results into the final, client-friendly format.

    This service enriches the raw output data by adding user-friendly timestamps
    and resolution-independent normalized coordinates for all bounding boxes.
    """

    def __init__(self, fps: float, video_width: int, video_height: int):
        """
        Initializes the formatter with the necessary video metadata.

        Args:
            fps: Frames per second of the source video.
            video_width: Width of the source video in pixels.
            video_height: Height of the source video in pixels.
        """
        self.fps = fps if fps > 0 else 30.0  # Default to 30 fps to avoid division by zero
        self.video_width = video_width
        self.video_height = video_height
        logger.info("OutputFormatterService initialized.")

    def _format_box(self, box_pixels: Dict[str, int]) -> Dict[str, float]:
        """Converts a single bounding box from absolute pixels to normalized coordinates."""
        if not all(k in box_pixels for k in ['x', 'y', 'width', 'height']):
            return {}
        
        return {
            "x_norm": box_pixels['x'] / self.video_width,
            "y_norm": box_pixels['y'] / self.video_height,
            "width_norm": box_pixels['width'] / self.video_width,
            "height_norm": box_pixels['height'] / self.video_height,
        }

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes a non-destructive transformation to format the pipeline's output.

        Args:
            data: The final hierarchical data from the ReframerPipeline.

        Returns:
            A deep copy of the data with added timestamps and normalized coordinates.
        """
        # Use a deep copy to ensure the transformation is non-destructive.
        formatted_data = copy.deepcopy(data)

        for scene in formatted_data:
            # Convert frame numbers to timestamps in seconds.
            scene["start_time_seconds"] = scene["start_frame"] / self.fps
            scene["end_time_seconds"] = scene["end_frame"] / self.fps

            if "detections" not in scene:
                continue

            for track_type in ["face_tracks", "person_tracks"]:
                for track in scene["detections"].get(track_type, []):
                    
                    if "fixed_box_pixels" in track:
                        track["fixed_box_normalized"] = self._format_box(track["fixed_box_pixels"])

                    if "dynamic_track" in track:
                        for frame_data in track["dynamic_track"]:
                            # Add timestamps and normalized boxes to each frame in a dynamic track.
                            frame_data["timestamp_seconds"] = frame_data["frame_number"] / self.fps
                            if "box_pixels" in frame_data:
                                frame_data["box_normalized"] = self._format_box(frame_data["box_pixels"])
        
        return formatted_data
