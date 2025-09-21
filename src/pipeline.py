# src/pipeline.py

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .services.face_detector_service import FaceDetectorService
from .services.face_track_processor_service import FaceTrackProcessorService
from .services.person_detector_service import PersonDetectorService
from .services.person_track_processor_service import PersonTrackProcessorService
from .services.scene_detector_service import SceneDetectorService
from .utils.caching import CacheManager


logger = logging.getLogger(__name__)


class ReframerPipeline:
    """
    Orchestrates the entire multi-stage computer vision pipeline for video reframing.

    This class manages the flow of data through a series of specialized services,
    handling everything from scene detection to final output formatting. It features
    an integrated caching system to make local runs fast and resumable.
    """

    def __init__(self, media_path: Path, config: Dict, cache_manager: CacheManager):
        self.source_media_path = media_path
        self.config = config
        self.cache_manager = cache_manager
        self._initialize_paths()
        self._initialize_data_holders()
        self._initialize_services()

    def _initialize_paths(self):
        """Defines all file paths for intermediate and final results."""
        project_name = self.source_media_path.stem
        self.project_data_path = Path("data") / project_name
        self.internal_media_path = self.project_data_path / self.source_media_path.name
        self.scene_list_path = self.project_data_path / f"{project_name}_scenes.json"
        self.scouting_results_path = self.project_data_path / f"{project_name}_scouting.json"
        self.face_tracks_path = self.project_data_path / f"{project_name}_face_tracks.json"
        self.person_tracks_path = self.project_data_path / f"{project_name}_person_tracks.json"
        self.final_output_path = self.project_data_path / f"{project_name}_final_output.json"

    def _initialize_data_holders(self):
        """Initializes instance variables to hold data between pipeline stages."""
        self.scene_list_data = None
        self.scouting_data = None
        self.face_tracks_data = None
        self.person_tracks_data = None
        self.final_formatted_data = None

    def _initialize_services(self):
        """Instantiates all the specialized services required by the pipeline."""
        self.scene_detector = SceneDetectorService(**self.config.get('scene_detector', {}))
        self.face_detector = FaceDetectorService(**self.config.get('face_detector', {}))
        self.person_detector = PersonDetectorService(**self.config.get('person_detector', {}))
        self.face_track_processor = FaceTrackProcessorService(**self.config.get('face_tracking', {}))
        self.person_track_processor = PersonTrackProcessorService(**self.config.get('person_tracking', {}))

    def _setup_project_directory(self):
        """Ensures the data directory for the project exists and copies the video."""
        self.project_data_path.mkdir(parents=True, exist_ok=True)
        if not self.internal_media_path.exists():
            shutil.copy(self.source_media_path, self.internal_media_path)

    def run(self) -> List[Dict[str, Any]]:
        """
        Executes the full pipeline in a sequential, step-by-step manner.

        Returns:
            A list of dictionaries representing the final, formatted output.
        """
        logger.info(f"--- Starting Reframer Pipeline for {self.source_media_path.name} ---")
        self._setup_project_directory()
        
        self._step_1_detect_scenes()
        self._step_2_run_scouting()
        self._step_3_process_tracks()
        
        logger.info(f"✅ Pipeline for {self.source_media_path.name} complete.")
        # Note: The final formatting is handled by the API/CLI entry points.
        return self.final_formatted_data

    def _step_1_detect_scenes(self):
        """Pipeline Step 1: Detect all scene changes in the video."""
        logger.info("Step 1: Detecting scenes...")
        cached_data = self.cache_manager.load(self.scene_list_path)
        if cached_data:
            self.scene_list_data = cached_data
        else:
            self.scene_list_data = self.scene_detector.run(media_path=self.internal_media_path)
            self.cache_manager.save(self.scene_list_path, self.scene_list_data)

    def _step_2_run_scouting(self):
        """Pipeline Step 2: Perform sparse keyframe sampling and object detection."""
        logger.info("Step 2: Running sparse scouting for faces and persons...")
        cached_data = self.cache_manager.load(self.scouting_results_path)
        if cached_data:
            self.scouting_data = cached_data
            return

        if not self.scene_list_data:
            self.scouting_data = {"faces": [], "persons": []}
            return

        # Calculate the number of frames to sample per scene based on config.
        scouting_cfg = self.config.get('scouting', {})
        min_samples = scouting_cfg.get('min_samples', 5)
        max_samples = scouting_cfg.get('max_samples', 20)
        sample_freq = scouting_cfg.get('sample_frequency', 1.0)
        
        all_scouting_results = {"faces": [], "persons": []}
        cap = cv2.VideoCapture(str(self.internal_media_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        for scene in self.scene_list_data:
            start_frame, end_frame = scene['start_frame'], scene['end_frame']
            duration_frames = end_frame - start_frame
            duration_seconds = duration_frames / fps if fps > 0 else 0
            
            num_samples = math.ceil(duration_seconds * sample_freq)
            final_sample_count = int(max(min_samples, min(num_samples, max_samples)))
            total_sampled_frames = min(final_sample_count, duration_frames + 1)
            
            frames_to_sample = np.linspace(start_frame, end_frame, num=total_sampled_frames, dtype=int)
            
            for frame_idx in frames_to_sample:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if not success: continue
                
                # Run detectors on the sampled frame.
                for face in self.face_detector.run(frame):
                    face['frame_number'] = int(frame_idx)
                    all_scouting_results['faces'].append(face)
                
                for person in self.person_detector.run(frame):
                    person['frame_number'] = int(frame_idx)
                    all_scouting_results['persons'].append(person)

        cap.release()
        self.scouting_data = all_scouting_results
        self.cache_manager.save(self.scouting_results_path, self.scouting_data)

    def _step_3_process_tracks(self):
        """Pipeline Step 3: Link sparse detections into persistent tracks."""
        logger.info("Step 3: Processing raw detections into face and person tracks...")
        if not self.scouting_data or not self.scene_list_data:
            self.face_tracks_data = []
            self.person_tracks_data = []
            return

        # --- Process Face Tracks ---
        cached_faces = self.cache_manager.load(self.face_tracks_path)
        if cached_faces:
            self.face_tracks_data = cached_faces
        else:
            raw_faces = self.scouting_data.get("faces", [])
            self.face_tracks_data = self._generate_tracks_for_type(raw_faces, self.face_track_processor)
            self.cache_manager.save(self.face_tracks_path, self.face_tracks_data)
        
        # --- Process Person Tracks ---
        cached_persons = self.cache_manager.load(self.person_tracks_path)
        if cached_persons:
            self.person_tracks_data = cached_persons
        else:
            raw_persons = self.scouting_data.get("persons", [])
            self.person_tracks_data = self._generate_tracks_for_type(raw_persons, self.person_track_processor)
            self.cache_manager.save(self.person_tracks_path, self.person_tracks_data)

    def _generate_tracks_for_type(self, raw_detections, track_processor):
        """Helper method to generate tracks for a specific detection type."""
        all_tracks = []
        output_modes = self.config.get('output', {}).get('modes', ['fixed'])
        
        for i, scene in enumerate(self.scene_list_data):
            start_frame, end_frame = scene['start_frame'], scene['end_frame']
            detections_in_scene = [d for d in raw_detections if start_frame <= d['frame_number'] <= end_frame]
            
            processed_tracks = track_processor.run(detections_in_scene, start_frame, end_frame, output_modes)
            for track in processed_tracks:
                track['scene_number'] = i + 1
            all_tracks.extend(processed_tracks)
        return all_tracks
