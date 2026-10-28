import sys
from types import SimpleNamespace

import numpy as np
import pytest

from facial_recognition.engine import compare_encodings, extract_faces


def test_compare_encodings_accepts_distance_at_threshold():
    result = compare_encodings(np.array([0.0, 0.0]), np.array([0.3, 0.4]), threshold=0.5)
    assert result.is_match is True
    assert result.distance == pytest.approx(0.5)
    assert result.threshold == 0.5


def test_compare_encodings_rejects_distance_above_threshold():
    result = compare_encodings(np.array([0.0, 0.0]), np.array([0.3, 0.4]), threshold=0.49)
    assert result.is_match is False
    assert result.distance == pytest.approx(0.5)


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf")])
def test_compare_encodings_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        compare_encodings(np.zeros(2), np.zeros(2), threshold)


def test_extract_faces_keeps_each_location_paired_with_its_encoding(monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    locations = [(1, 10, 10, 1), (2, 19, 12, 11)]
    encodings = [np.ones(128), np.zeros(128)]
    monkeypatch.setitem(sys.modules, "face_recognition", SimpleNamespace(
        face_locations=lambda rgb, model: locations,
        face_encodings=lambda rgb, known_face_locations: encodings,
    ))

    faces = extract_faces(image)

    assert [face.location for face in faces] == locations
    np.testing.assert_array_equal(faces[0].encoding, encodings[0])
    np.testing.assert_array_equal(faces[1].encoding, encodings[1])


def test_extract_faces_returns_empty_when_none_detected(monkeypatch):
    monkeypatch.setitem(sys.modules, "face_recognition", SimpleNamespace(
        face_locations=lambda rgb, model: [],
        face_encodings=lambda rgb, known_face_locations: [],
    ))
    assert extract_faces(np.zeros((20, 20, 3), dtype=np.uint8)) == []


def test_extract_faces_reports_missing_encoding(monkeypatch):
    monkeypatch.setitem(sys.modules, "face_recognition", SimpleNamespace(
        face_locations=lambda rgb, model: [(1, 10, 10, 1)],
        face_encodings=lambda rgb, known_face_locations: [],
    ))
    with pytest.raises(ValueError, match="encoding"):
        extract_faces(np.zeros((20, 20, 3), dtype=np.uint8))
