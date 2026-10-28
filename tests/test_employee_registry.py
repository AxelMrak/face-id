import sys
from types import SimpleNamespace

import numpy as np
import pytest

from facial_recognition.employee_registry import load_employees
from facial_recognition.models import DetectedFace


def make_images(directory, *names):
    for name in names:
        (directory / name).write_bytes(b"image placeholder")


def stub_images(monkeypatch, *, faces=None, errors=None):
    """Keep filesystem discovery real; replace only unavailable photo decoding/detection."""
    faces = faces or {}
    errors = errors or {}

    def load_image_file(path):
        if path.name in errors:
            raise OSError("unreadable image")
        return np.array([path.name])

    monkeypatch.setitem(sys.modules, "face_recognition", SimpleNamespace(load_image_file=load_image_file))
    monkeypatch.setattr(
        "facial_recognition.employee_registry.extract_faces",
        lambda image: faces.get(image[0], [DetectedFace((0, 1, 1, 0), np.ones(128))]),
    )


def test_loads_three_employees_with_stable_ids_and_names(tmp_path, monkeypatch):
    make_images(tmp_path, "EMP003__Maria_Lopez.png", "EMP001__Ana_Garcia.jpg", "EMP002__Juan_Perez.jpeg")
    stub_images(monkeypatch)

    employees = load_employees(str(tmp_path))

    assert [(employee.id, employee.name) for employee in employees] == [
        ("EMP001", "Ana Garcia"), ("EMP002", "Juan Perez"), ("EMP003", "Maria Lopez")
    ]
    np.testing.assert_array_equal(employees[0].encoding, np.ones(128))


@pytest.mark.parametrize(
    ("invalid_name", "invalid_faces", "error", "message"),
    [
        ("EMP002__No_Face.jpg", [], None, "exactly one face"),
        ("EMP002__Group.jpg", [DetectedFace((0, 1, 1, 0), np.ones(128))] * 2, None, "exactly one face"),
        ("EMP002__Broken.jpg", None, True, "unreadable image"),
    ],
)
def test_skips_bad_photo_and_warns(tmp_path, monkeypatch, invalid_name, invalid_faces, error, message):
    make_images(tmp_path, "EMP001__Ana.jpg", invalid_name)
    stub_images(
        monkeypatch,
        faces={invalid_name: invalid_faces} if invalid_faces is not None else {},
        errors={invalid_name: True} if error else {},
    )

    with pytest.warns(RuntimeWarning, match=message):
        employees = load_employees(str(tmp_path))

    assert [employee.id for employee in employees] == ["EMP001"]


def test_rejects_duplicate_ids_instead_of_choosing_one(tmp_path, monkeypatch):
    make_images(tmp_path, "EMP001__Ana.jpg", "EMP001__Otra.png")
    stub_images(monkeypatch)

    with pytest.raises(ValueError, match="EMP001"):
        load_employees(str(tmp_path))


def test_skips_invalid_filename_and_warns(tmp_path, monkeypatch):
    make_images(tmp_path, "bad-name.jpg", "EMP001__Ana.jpg")
    stub_images(monkeypatch)

    with pytest.warns(RuntimeWarning, match="bad-name.jpg"):
        employees = load_employees(str(tmp_path))

    assert [employee.id for employee in employees] == ["EMP001"]


def test_rejects_empty_or_all_invalid_directory(tmp_path, monkeypatch):
    stub_images(monkeypatch)
    with pytest.raises(ValueError, match="no valid employees"):
        load_employees(str(tmp_path))

    make_images(tmp_path, "EMP001__Empty.jpg")
    stub_images(monkeypatch, faces={"EMP001__Empty.jpg": []})
    with pytest.warns(RuntimeWarning, match="exactly one face"), pytest.raises(
        ValueError, match="no valid employees"
    ):
        load_employees(str(tmp_path))


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_employees(str(tmp_path / "missing"))
