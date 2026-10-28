"""Local integration checks using private photos excluded from Git."""

from pathlib import Path

import pytest

from facial_recognition.employee_registry import load_employees
from facial_recognition.engine import extract_faces, find_best_match

ROOT = Path(__file__).parent.parent
REGISTRY = ROOT / "data" / "employees"
QUERIES = ROOT / "data" / "test-images"


@pytest.fixture(scope="module")
def employees():
    if not list(REGISTRY.glob("*.jp*g")):
        pytest.skip("local employee photos are not available")
    return load_employees(str(REGISTRY))


@pytest.mark.parametrize(
    ("filename", "expected_id"),
    [
        ("alberto-2.jpeg", "AR001"),
        ("delarua-2.jpeg", "AR002"),
        ("menem-2.jpeg", "AR003"),
    ],
)
def test_identifies_president_from_a_different_photo(employees, filename, expected_id):
    image = QUERIES / filename
    if not image.is_file():
        pytest.skip(f"local query photo missing: {filename}")

    import face_recognition

    faces = extract_faces(face_recognition.load_image_file(image))
    assert len(faces) == 1
    result = find_best_match(faces[0].encoding, employees, threshold=0.6)
    assert result.employee_id == expected_id
    assert result.distance is not None and result.distance <= result.threshold


def test_marks_unregistered_person_as_unknown(employees):
    image = ROOT / "demo-images" / "trump-1.jpeg"
    if not image.is_file():
        pytest.skip("local unknown-person photo is not available")

    import face_recognition

    faces = extract_faces(face_recognition.load_image_file(image))
    assert len(faces) == 1
    result = find_best_match(faces[0].encoding, employees, threshold=0.6)
    assert result.employee_id is None
    assert result.distance is not None and result.distance > result.threshold
