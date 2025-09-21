# src/services/person_track_processor_service.py

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from ..utils.bounding_box import calculate_fixed_box

logger = logging.getLogger(__name__)


class PersonTrackProcessorService:
    """
    Processes sparse person detections from a scene into stable, persistent,
    and fully interpolated/extrapolated tracks suitable for reframing.
    """

    def __init__(self, distance_threshold_factor: float = 0.7, min_persistence_ratio: float = 0.4):
        """
        Initializes the track processor with key algorithmic parameters.

        Args:
            distance_threshold_factor: Max distance a person's center can move
                (as a factor of their width) between frames to be linked to a track.
            min_persistence_ratio: A track is kept if it appears in at least this
                ratio of the total sampled frames.
        """
        self.distance_threshold_factor = distance_threshold_factor
        self.min_persistence_ratio = min_persistence_ratio
        logger.info(f"PersonTrackProcessorService initialized.")

    def run(self, sparse_detections: List[Dict], scene_start_frame: int, scene_end_frame: int, total_sampled_frames: int, output_modes: List[str]) -> List[Dict[str, Any]]:
        """Orchestrates the full tracking, filtering, and processing logic for persons."""
        if not sparse_detections:
            return []

        detections_by_frame = self._group_detections_by_frame(sparse_detections)
        raw_tracks = self._link_detections_into_tracks(detections_by_frame)
        
        min_hits = np.ceil(total_sampled_frames * self.min_persistence_ratio)
        persistent_tracks = self._filter_tracks(raw_tracks, min_hits)
        logger.info(f"Found {len(persistent_tracks)} persistent person tracks (min_hits: {int(min_hits)}).")

        final_tracks = self._process_final_tracks(persistent_tracks, scene_start_frame, scene_end_frame, output_modes)
        return final_tracks
    
    def _group_detections_by_frame(self, detections: List[Dict]) -> Dict[int, List[Dict]]:
        """Groups a flat list of detections into a dictionary keyed by frame number."""
        grouped = defaultdict(list)
        for det in detections:
            grouped[det['frame_number']].append(det)
        return dict(sorted(grouped.items()))

    def _link_detections_into_tracks(self, detections_by_frame: Dict[int, List[Dict]]) -> List[Dict]:
        """Core tracking algorithm that links sparse detections into continuous tracks."""
        active_tracks = {}
        next_track_id = 0
        sorted_frames = list(detections_by_frame.keys())

        for frame_num in sorted_frames:
            current_detections = detections_by_frame.get(frame_num, [])
            if not current_detections: continue

            det_centers = np.array([(d['box_pixels']['x'] + d['box_pixels']['width'] / 2, d['box_pixels']['y'] + d['box_pixels']['height'] / 2) for d in current_detections])
            det_sizes = np.array([float(d['box_pixels']['width']) for d in current_detections])
            
            if active_tracks:
                track_ids = list(active_tracks.keys())
                track_centers = np.array([track['last_center'] for track in active_tracks.values()])
                track_sizes = np.array([track['last_size'] for track in active_tracks.values()])
                
                distances = cdist(track_centers, det_centers)
                
                for i, track_id in enumerate(track_ids):
                    track_max_dist = track_sizes[i] * self.distance_threshold_factor
                    possible_matches = np.where(distances[i, :] < track_max_dist)[0]
                    
                    best_dist, best_det_idx = float('inf'), -1
                    for det_idx in possible_matches:
                        if distances[i, det_idx] < best_dist and not any(t.get('match_idx') == det_idx for t in active_tracks.values()):
                            best_dist, best_det_idx = distances[i, det_idx], det_idx
                    
                    if best_det_idx != -1:
                        track = active_tracks[track_id]
                        track['detections'].append(current_detections[best_det_idx])
                        track['last_center'] = det_centers[best_det_idx]
                        track['last_size'] = det_sizes[best_det_idx]
                        track['match_idx'] = best_det_idx
            
            matched_indices = {t['match_idx'] for t in active_tracks.values() if 'match_idx' in t}
            for det_idx in set(range(len(current_detections))) - matched_indices:
                new_track_id = f"PERSON_{next_track_id}"
                active_tracks[new_track_id] = {
                    'detections': [current_detections[det_idx]],
                    'last_center': det_centers[det_idx],
                    'last_size': det_sizes[det_idx],
                }
                next_track_id += 1

            for track in active_tracks.values():
                track.pop('match_idx', None)

        return list(active_tracks.values())

    def _filter_tracks(self, tracks: List[Dict], min_hits: int) -> List[Dict]:
        """Filters out noisy tracks that do not appear in enough sampled keyframes."""
        return [track for track in tracks if len(track['detections']) >= min_hits]

    def _process_final_tracks(self, tracks: List[Dict], start_frame: int, end_frame: int, output_modes: List[str]) -> List[Dict]:
        """Generates the final 'fixed' and/or 'dynamic' track outputs."""
        final_tracks = []
        for i, track in enumerate(tracks):
            track_output = {"track_id": f"PERSON_{i+1}"}
            measured_detections = sorted(track['detections'], key=lambda x: x['frame_number'])
            
            if 'fixed' in output_modes:
                measured_boxes = [det['box_pixels'] for det in measured_detections]
                track_output['fixed_box_pixels'] = calculate_fixed_box(measured_boxes)
            
            if 'dynamic' in output_modes:
                dynamic_track = []
                
                # Interpolate between measured keyframes.
                for j in range(len(measured_detections) - 1):
                    start_det, end_det = measured_detections[j], measured_detections[j+1]
                    start_box = np.array(list(start_det['box_pixels'].values()))
                    end_box = np.array(list(end_det['box_pixels'].values()))
                    for frame_num in range(start_det['frame_number'], end_det['frame_number']):
                        alpha = (frame_num - start_det['frame_number']) / (end_det['frame_number'] - start_det['frame_number'])
                        inter_box_coords = (1 - alpha) * start_box + alpha * end_box
                        dynamic_track.append({
                            "frame_number": frame_num,
                            "box_pixels": dict(zip(['x', 'y', 'width', 'height'], map(int, inter_box_coords)))
                        })
                dynamic_track.append(measured_detections[-1])

                # Extrapolate to the start and end of the scene.
                first_det, last_det = measured_detections[0], measured_detections[-1]
                for frame_num in range(start_frame, first_det['frame_number']):
                    dynamic_track.append({"frame_number": frame_num, "box_pixels": first_det['box_pixels']})
                for frame_num in range(last_det['frame_number'] + 1, end_frame + 1):
                    dynamic_track.append({"frame_number": frame_num, "box_pixels": last_det['box_pixels']})

                track_output['dynamic_track'] = sorted(dynamic_track, key=lambda x: x['frame_number'])

            final_tracks.append(track_output)
        return final_tracks
