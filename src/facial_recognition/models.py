"""Result types shared by the engine and its consumers."""

from dataclasses import dataclass

import numpy as np

FaceLocation = tuple[int, int, int, int]  # top, right, bottom, left


@dataclass(frozen=True)
class DetectedFace:
    location: FaceLocation
    encoding: np.ndarray


@dataclass(frozen=True)
class ComparisonResult:
    is_match: bool
    distance: float
    threshold: float


@dataclass(frozen=True)
class Employee:
    id: str
    name: str
    encoding: np.ndarray


@dataclass(frozen=True)
class IdentificationResult:
    employee_id: str | None
    distance: float | None
    threshold: float


@dataclass(frozen=True)
class RecognizedFace:
    location: FaceLocation
    employee_id: str | None
    distance: float | None
