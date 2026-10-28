import numpy as np
import pytest

from facial_recognition.models import DetectedFace, Employee
from scripts import attendance


def registered_employee():
    return Employee("EMP001", "Ana Garcia", np.zeros(128))


def test_single_registered_face_is_labeled_in_green(monkeypatch):
    seen_rgb = []
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    frame[40, 40] = (1, 2, 3)  # BGR

    def extract(rgb):
        seen_rgb.append(tuple(rgb[40, 40]))
        return [DetectedFace((20, 40, 40, 20), np.zeros(128))]

    monkeypatch.setattr(attendance, "extract_faces", extract)
    labels = []
    monkeypatch.setattr(attendance.cv2, "putText", lambda image, text, *args: labels.append(text))
    output = attendance.annotate_frame(frame, [registered_employee()], 0.6)
    assert labels == ["Ana Garcia"]
    assert seen_rgb == [(3, 2, 1)]
    assert tuple(output[20, 20]) == (0, 255, 0)  # green rectangle in BGR
    assert not frame[20, 20].any()  # original frame is not modified


def test_unknown_face_is_labeled_in_red(monkeypatch):
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: [
        DetectedFace((20, 40, 40, 20), np.ones(128))
    ])
    labels = []
    monkeypatch.setattr(attendance.cv2, "putText", lambda image, text, *args: labels.append(text))
    output = attendance.annotate_frame(np.zeros((60, 60, 3), dtype=np.uint8), [registered_employee()], 0.6)
    assert labels == ["UNKNOWN"]
    assert tuple(output[20, 20]) == (0, 0, 255)


def test_multiple_faces_are_marked_without_identifying_anyone(monkeypatch):
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: [
        DetectedFace((20, 40, 40, 20), np.zeros(128)),
        DetectedFace((30, 50, 50, 30), np.ones(128)),
    ])
    monkeypatch.setattr(attendance, "find_best_match", lambda *args: pytest.fail("must not identify multiple faces"))
    labels = []
    monkeypatch.setattr(attendance.cv2, "putText", lambda image, text, *args: labels.append(text))
    output = attendance.annotate_frame(np.zeros((80, 80, 3), dtype=np.uint8), [registered_employee()], 0.6)
    assert labels == ["MULTIPLE_FACES"]
    assert tuple(output[20, 20]) == (0, 255, 255)
    assert tuple(output[30, 30]) == (0, 255, 255)


def test_no_faces_keeps_frame_unannotated(monkeypatch):
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: [])
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    np.testing.assert_array_equal(attendance.annotate_frame(frame, [registered_employee()], 0.6), frame)


def test_main_releases_camera_when_user_quits(monkeypatch):
    capture = object()
    released = []
    monkeypatch.setattr(attendance, "load_employees", lambda directory: [registered_employee()])
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: capture)
    monkeypatch.setattr(attendance.camera, "read_frame", lambda device: np.zeros((60, 60, 3), dtype=np.uint8))
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: [])
    monkeypatch.setattr(attendance.camera, "show_frame", lambda frame: True)
    monkeypatch.setattr(attendance.camera, "close_camera", lambda device: released.append(device))

    assert attendance.main(["--camera", "1", "--threshold", "0.5"]) == 0
    assert released == [capture]


def test_main_releases_camera_on_capture_failure(monkeypatch, capsys):
    capture = object()
    released = []
    monkeypatch.setattr(attendance, "load_employees", lambda directory: [registered_employee()])
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: capture)
    monkeypatch.setattr(attendance.camera, "read_frame", lambda device: (_ for _ in ()).throw(RuntimeError("capture failed")))
    monkeypatch.setattr(attendance.camera, "close_camera", lambda device: released.append(device))

    assert attendance.main([]) == 2
    assert released == [capture]
    assert "capture failed" in capsys.readouterr().err


def test_main_releases_camera_on_display_failure(monkeypatch, capsys):
    capture = object()
    released = []
    monkeypatch.setattr(attendance, "load_employees", lambda directory: [registered_employee()])
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: capture)
    monkeypatch.setattr(attendance.camera, "read_frame", lambda device: np.zeros((60, 60, 3), dtype=np.uint8))
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: [])
    monkeypatch.setattr(attendance.camera, "show_frame", lambda frame: (_ for _ in ()).throw(RuntimeError("display failed")))
    monkeypatch.setattr(attendance.camera, "close_camera", lambda device: released.append(device))

    assert attendance.main([]) == 2
    assert released == [capture]
    assert "display failed" in capsys.readouterr().err


def test_main_reports_missing_employee_directory_before_opening_camera(monkeypatch, capsys):
    monkeypatch.setattr(attendance, "load_employees", lambda directory: (_ for _ in ()).throw(FileNotFoundError(directory)))
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: pytest.fail("camera should remain closed"))
    assert attendance.main(["--employees", "missing"]) == 2
    assert "missing" in capsys.readouterr().err
