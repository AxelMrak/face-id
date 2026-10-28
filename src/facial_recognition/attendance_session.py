"""Shared frame recognition and attendance coordination for CLI and desktop UI."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from facial_recognition.attendance_service import AttendanceTracker
from facial_recognition.engine import extract_faces, find_best_match
from facial_recognition.models import (AttendanceResult, DetectedFace,
                                       Employee, IdentificationResult)


@dataclass(frozen=True)
class FrameObservation:
    faces: list[DetectedFace]
    employee: Employee | None
    invalid_encoding: bool = False
    attendance: AttendanceResult | None = None


def recognize_frame(
    frame_bgr: np.ndarray,
    employees: list[Employee],
    threshold: float,
    *,
    extractor: Callable[[np.ndarray], list[DetectedFace]] = extract_faces,
    matcher: Callable[..., IdentificationResult] = find_best_match,
) -> FrameObservation:
    """Identify only a solitary face; reject invalid encodings as empty frames."""
    try:
        faces = extractor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    except ValueError:
        return FrameObservation([], None, invalid_encoding=True)
    employee = None
    if len(faces) == 1:
        match = matcher(faces[0].encoding, employees, threshold=threshold)
        employee = next((person for person in employees if person.id == match.employee_id), None)
    return FrameObservation(faces, employee)


class AttendanceSession:
    """Keep recognition and confirmation together without owning camera or UI."""

    def __init__(self, employees: list[Employee], threshold: float, tracker: AttendanceTracker):
        self.employees = employees
        self.threshold = threshold
        self.tracker = tracker
        self.extractor: Callable[[np.ndarray], list[DetectedFace]] = extract_faces

    def reset(self, at: datetime) -> None:
        self.tracker.observe(None, at)

    def process(self, frame_bgr: np.ndarray, at: datetime) -> FrameObservation:
        observation = recognize_frame(
            frame_bgr, self.employees, self.threshold, extractor=self.extractor
        )
        attendance = self.tracker.observe(observation.employee, at)
        return FrameObservation(
            observation.faces, observation.employee, observation.invalid_encoding, attendance
        )
