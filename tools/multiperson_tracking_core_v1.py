"""Small, profile-free track association core for the multi-person pilot.

The association deliberately uses only image geometry. Limb lengths and
subject profiles are experimental endpoints, so using them here would create
circular evidence in the body-profile identification experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PoseDetection:
    """One frame-local pose detection in normalised image coordinates."""

    instance_id: int
    centre_x: float
    centre_y: float
    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float


@dataclass
class TrackState:
    """Minimal state retained between frames."""

    track_id: int
    centre_x: float
    centre_y: float
    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float
    last_frame_index: int


def bbox_iou(first: PoseDetection | TrackState, second: PoseDetection) -> float:
    left = max(float(first.bbox_left), float(second.bbox_left))
    top = max(float(first.bbox_top), float(second.bbox_top))
    right = min(float(first.bbox_right), float(second.bbox_right))
    bottom = min(float(first.bbox_bottom), float(second.bbox_bottom))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first.bbox_right) - float(first.bbox_left)) * max(
        0.0, float(first.bbox_bottom) - float(first.bbox_top)
    )
    second_area = max(
        0.0, float(second.bbox_right) - float(second.bbox_left)
    ) * max(0.0, float(second.bbox_bottom) - float(second.bbox_top))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def association_cost(
    track: TrackState,
    detection: PoseDetection,
    max_centre_distance: float,
) -> float | None:
    """Return a bounded geometry cost, or ``None`` for an impossible match."""

    distance = math.hypot(
        float(track.centre_x) - float(detection.centre_x),
        float(track.centre_y) - float(detection.centre_y),
    )
    if distance > float(max_centre_distance):
        return None
    distance_term = distance / float(max_centre_distance)
    overlap_term = 1.0 - bbox_iou(track, detection)
    return 0.8 * distance_term + 0.2 * overlap_term


def optimal_assignment(
    tracks: list[TrackState],
    detections: list[PoseDetection],
    max_centre_distance: float,
) -> list[tuple[int, int, float]]:
    """Maximise valid matches, then minimise total cost.

    The registered pilot has at most two people and the stress test at most
    three. Exhaustive search is therefore clearer and more reproducible than a
    new SciPy dependency, and it naturally supports unmatched tracks.
    """

    best_key: tuple[int, float, tuple[tuple[int, int], ...]] | None = None
    best_pairs: list[tuple[int, int, float]] = []

    def visit(
        track_index: int,
        used_detection_indices: set[int],
        pairs: list[tuple[int, int, float]],
    ) -> None:
        nonlocal best_key, best_pairs
        if track_index >= len(tracks):
            stable_pairs = tuple(
                sorted((track_id, detection_index) for track_id, detection_index, _ in pairs)
            )
            key = (-len(pairs), sum(cost for _, _, cost in pairs), stable_pairs)
            if best_key is None or key < best_key:
                best_key = key
                best_pairs = list(pairs)
            return

        visit(track_index + 1, used_detection_indices, pairs)
        track = tracks[track_index]
        for detection_index, detection in enumerate(detections):
            if detection_index in used_detection_indices:
                continue
            cost = association_cost(track, detection, max_centre_distance)
            if cost is None:
                continue
            used_detection_indices.add(detection_index)
            pairs.append((track.track_id, detection_index, cost))
            visit(track_index + 1, used_detection_indices, pairs)
            pairs.pop()
            used_detection_indices.remove(detection_index)

    visit(0, set(), [])
    return sorted(best_pairs)


class TrackManager:
    """Assign deterministic track IDs without using body-profile features."""

    def __init__(
        self,
        *,
        max_centre_distance: float = 0.20,
        max_missed_frames: int = 15,
    ) -> None:
        if max_centre_distance <= 0.0:
            raise ValueError("max_centre_distance must be positive")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        self.max_centre_distance = float(max_centre_distance)
        self.max_missed_frames = int(max_missed_frames)
        self.next_track_id = 0
        self.tracks: dict[int, TrackState] = {}

    def _expire(self, frame_index: int) -> None:
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if int(frame_index) - int(track.last_frame_index)
            > self.max_missed_frames
        ]
        for track_id in expired:
            del self.tracks[track_id]

    def update(
        self,
        frame_index: int,
        detections: list[PoseDetection],
    ) -> list[dict]:
        """Return association metadata in the original detection order."""

        frame_index = int(frame_index)
        self._expire(frame_index)
        active_tracks = [self.tracks[key] for key in sorted(self.tracks)]
        assignments = optimal_assignment(
            active_tracks,
            detections,
            self.max_centre_distance,
        )
        by_detection: dict[int, dict] = {}
        for track_id, detection_index, cost in assignments:
            detection = detections[detection_index]
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                centre_x=detection.centre_x,
                centre_y=detection.centre_y,
                bbox_left=detection.bbox_left,
                bbox_top=detection.bbox_top,
                bbox_right=detection.bbox_right,
                bbox_bottom=detection.bbox_bottom,
                last_frame_index=frame_index,
            )
            by_detection[detection_index] = {
                "track_id": track_id,
                "association_status": "matched",
                "association_cost": cost,
                "association_confidence": max(0.0, min(1.0, 1.0 - cost)),
            }

        for detection_index, detection in enumerate(detections):
            if detection_index in by_detection:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                centre_x=detection.centre_x,
                centre_y=detection.centre_y,
                bbox_left=detection.bbox_left,
                bbox_top=detection.bbox_top,
                bbox_right=detection.bbox_right,
                bbox_bottom=detection.bbox_bottom,
                last_frame_index=frame_index,
            )
            by_detection[detection_index] = {
                "track_id": track_id,
                "association_status": "new",
                "association_cost": None,
                "association_confidence": 0.0,
            }

        return [by_detection[index] for index in range(len(detections))]

    def active_track_count(self, frame_index: int) -> int:
        self._expire(int(frame_index))
        return len(self.tracks)
