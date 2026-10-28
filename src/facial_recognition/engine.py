"""Detection and comparison of faces in RGB images."""

import numpy as np

from facial_recognition.models import ComparisonResult, DetectedFace


def extract_faces(image_rgb: np.ndarray) -> list[DetectedFace]:
    """Detect all faces in an RGB image using the CPU HOG detector."""
    import face_recognition

    locations = face_recognition.face_locations(image_rgb, model="hog")
    if not locations:
        return []
    encodings = face_recognition.face_encodings(image_rgb, known_face_locations=locations)
    if len(encodings) != len(locations) or any(
        encoding is None or np.shape(encoding) != (128,) or not np.all(np.isfinite(encoding))
        for encoding in encodings
    ):
        raise ValueError("A detected face has no valid encoding")
    return [DetectedFace(location, encoding) for location, encoding in zip(locations, encodings, strict=True)]


def compare_encodings(
    first: np.ndarray, second: np.ndarray, threshold: float = 0.6
) -> ComparisonResult:
    """Compare two encodings using Euclidean distance (inclusive threshold)."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 1 or first.shape != second.shape or not (
        np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    ):
        raise ValueError("encodings must be finite vectors of the same size")
    distance = float(np.linalg.norm(first - second))
    return ComparisonResult(distance <= threshold, distance, threshold)
