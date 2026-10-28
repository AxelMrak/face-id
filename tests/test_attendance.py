from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from facial_recognition.attendance_csv import AttendanceCSV
from facial_recognition.attendance_service import (AttendanceService,
                                                   AttendanceTracker)
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


def test_invalid_encoding_skips_frame_and_resets_confirmation(tmp_path, monkeypatch, capsys):
    path = tmp_path / "attendance.csv"
    tracker = AttendanceTracker(AttendanceService(AttendanceCSV(path)))
    when = datetime(2026, 10, 28, 8, 5, tzinfo=timezone.utc)
    face = DetectedFace((20, 40, 40, 20), np.zeros(128))
    frames = iter([[face], [face], ValueError("invalid encoding"), [face], [face], [face]])

    def extract(rgb):
        item = next(frames)
        if isinstance(item, ValueError):
            raise item
        return item

    monkeypatch.setattr(attendance, "extract_faces", extract)
    for _ in range(5):
        attendance.annotate_frame(
            np.zeros((60, 60, 3), dtype=np.uint8), [registered_employee()], 0.6,
            on_match=lambda employee: tracker.observe(employee, when),
        )
    assert not path.exists()
    attendance.annotate_frame(
        np.zeros((60, 60, 3), dtype=np.uint8), [registered_employee()], 0.6,
        on_match=lambda employee: tracker.observe(employee, when),
    )
    assert len(AttendanceCSV(path).load_records()) == 1
    assert "invalid encoding" in capsys.readouterr().err


def test_three_matching_frames_register_and_show_status_without_duplicate(tmp_path, monkeypatch):
    path = tmp_path / "attendance.csv"
    tracker = AttendanceTracker(AttendanceService(AttendanceCSV(path)))
    when = datetime(2026, 10, 28, 8, 5, tzinfo=timezone(timedelta(hours=-3)))
    face = DetectedFace((20, 40, 40, 20), np.zeros(128))
    frames = iter([[face], [face], [], [face], [face], [face], [face], [face], [face], [face]])
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: next(frames))
    labels = []
    monkeypatch.setattr(attendance.cv2, "putText", lambda image, text, *args: labels.append(text))
    for index in range(7):
        attendance.annotate_frame(
            np.zeros((60, 60, 3), dtype=np.uint8), [registered_employee()], 0.6,
            on_match=lambda employee: tracker.observe(employee, when),
        )
        if index < 5:
            assert not path.exists()
    assert len(AttendanceCSV(path).load_records()) == 1
    assert any("Attendance registered" in label for label in labels)

    tracker.observe(None, when)
    for _ in range(3):
        attendance.annotate_frame(
            np.zeros((60, 60, 3), dtype=np.uint8), [registered_employee()], 0.6,
            on_match=lambda employee: tracker.observe(employee, when),
        )
    assert any("Already registered today" in label for label in labels)
    assert len(AttendanceCSV(path).load_records()) == 1


def test_unknown_and_multiple_faces_never_register(tmp_path, monkeypatch):
    path = tmp_path / "attendance.csv"
    tracker = AttendanceTracker(AttendanceService(AttendanceCSV(path)))
    when = datetime(2026, 10, 28, 8, 5, tzinfo=timezone.utc)
    unknown = DetectedFace((20, 40, 40, 20), np.ones(128))
    known = DetectedFace((20, 40, 40, 20), np.zeros(128))
    frames = iter([[unknown], [unknown], [known, unknown], [unknown]])
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: next(frames))
    for _ in range(4):
        attendance.annotate_frame(
            np.zeros((60, 60, 3), dtype=np.uint8), [registered_employee()], 0.6,
            on_match=lambda employee: tracker.observe(employee, when),
        )
    assert not path.exists()


def test_main_releases_camera_when_user_quits(monkeypatch, tmp_path):
    capture = object()
    released = []
    monkeypatch.setattr(attendance, "load_employees", lambda directory: [registered_employee()])
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: capture)
    monkeypatch.setattr(attendance.camera, "read_frame", lambda device: np.zeros((60, 60, 3), dtype=np.uint8))
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: [])
    monkeypatch.setattr(attendance.camera, "show_frame", lambda frame: True)
    monkeypatch.setattr(attendance.camera, "close_camera", lambda device: released.append(device))

    assert attendance.main(["--camera", "1", "--threshold", "0.5", "--attendance-csv", str(tmp_path / "attendance.csv")]) == 0
    assert released == [capture]


def test_main_retries_transient_capture_and_resets_confirmation(monkeypatch, tmp_path):
    path = tmp_path / "attendance.csv"
    capture = object()
    released = []
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    face = DetectedFace((20, 40, 40, 20), np.zeros(128))
    reads = iter([frame, frame, RuntimeError("capture failed"), frame, frame, frame])
    shown = []

    def read(device):
        item = next(reads)
        if isinstance(item, RuntimeError):
            raise item
        return item

    def show(image):
        shown.append(image)
        if len(shown) == 4:
            assert not path.exists()
        return len(shown) == 5

    monkeypatch.setattr(attendance, "load_employees", lambda directory: [registered_employee()])
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: capture)
    monkeypatch.setattr(attendance.camera, "read_frame", read)
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: [face])
    monkeypatch.setattr(attendance.camera, "show_frame", show)
    monkeypatch.setattr(attendance.camera, "close_camera", lambda device: released.append(device))

    assert attendance.main(["--attendance-csv", str(path)]) == 0
    assert len(AttendanceCSV(path).load_records()) == 1
    assert released == [capture]


def test_main_releases_camera_on_capture_failure(monkeypatch, capsys, tmp_path):
    capture = object()
    released = []
    attempts = []
    monkeypatch.setattr(attendance, "load_employees", lambda directory: [registered_employee()])
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: capture)

    def fail(device):
        attempts.append(device)
        raise RuntimeError("capture failed")

    monkeypatch.setattr(attendance.camera, "read_frame", fail)
    monkeypatch.setattr(attendance.camera, "close_camera", lambda device: released.append(device))

    assert attendance.main(["--attendance-csv", str(tmp_path / "attendance.csv")]) == 2
    assert attempts == [capture] * 3
    assert released == [capture]
    assert "capture failed" in capsys.readouterr().err


def test_main_releases_camera_on_display_failure(monkeypatch, capsys, tmp_path):
    capture = object()
    released = []
    monkeypatch.setattr(attendance, "load_employees", lambda directory: [registered_employee()])
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: capture)
    monkeypatch.setattr(attendance.camera, "read_frame", lambda device: np.zeros((60, 60, 3), dtype=np.uint8))
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: [])
    monkeypatch.setattr(attendance.camera, "show_frame", lambda frame: (_ for _ in ()).throw(RuntimeError("display failed")))
    monkeypatch.setattr(attendance.camera, "close_camera", lambda device: released.append(device))

    assert attendance.main(["--attendance-csv", str(tmp_path / "attendance.csv")]) == 2
    assert released == [capture]
    assert "display failed" in capsys.readouterr().err


def test_main_registers_after_three_frames_and_keeps_history_on_restart(tmp_path, monkeypatch):
    path = tmp_path / "attendance.csv"
    capture = object()
    shown = []
    face = DetectedFace((20, 40, 40, 20), np.zeros(128))
    monkeypatch.setattr(attendance, "load_employees", lambda directory: [registered_employee()])
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: capture)
    monkeypatch.setattr(attendance.camera, "read_frame", lambda device: np.zeros((60, 60, 3), dtype=np.uint8))
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: [face])
    monkeypatch.setattr(attendance.camera, "show_frame", lambda frame: shown.append(frame) or len(shown) % 3 == 0)
    monkeypatch.setattr(attendance.camera, "close_camera", lambda device: None)

    assert attendance.main(["--attendance-csv", str(path)]) == 0
    assert len(shown) == 3
    first = AttendanceCSV(path).load_records()
    assert len(first) == 1
    assert first[0].employee_id == "EMP001"
    assert first[0].check_in_at.utcoffset() is not None

    assert attendance.main(["--attendance-csv", str(path)]) == 0
    assert len(AttendanceCSV(path).load_records()) == 1


def test_main_write_failure_releases_camera_without_confirmation(tmp_path, monkeypatch, capsys):
    path = tmp_path / "attendance.csv"
    capture = object()
    released = []
    labels = []
    face = DetectedFace((20, 40, 40, 20), np.zeros(128))
    monkeypatch.setattr(attendance, "load_employees", lambda directory: [registered_employee()])
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: capture)
    monkeypatch.setattr(attendance.camera, "read_frame", lambda device: np.zeros((60, 60, 3), dtype=np.uint8))
    monkeypatch.setattr(attendance, "extract_faces", lambda rgb: [face])
    monkeypatch.setattr(attendance.cv2, "putText", lambda image, text, *args: labels.append(text))
    monkeypatch.setattr(attendance.camera, "show_frame", lambda frame: False)
    monkeypatch.setattr(attendance.camera, "close_camera", lambda device: released.append(device))
    monkeypatch.setattr(AttendanceCSV, "append_record", lambda self, record: (_ for _ in ()).throw(OSError("disk full")))

    assert attendance.main(["--attendance-csv", str(path)]) == 2
    assert released == [capture]
    assert "disk full" in capsys.readouterr().err
    assert "Attendance registered" not in labels
    assert not path.exists()


def test_main_rejects_malformed_csv_before_opening_camera(tmp_path, monkeypatch, capsys):
    path = tmp_path / "attendance.csv"
    path.write_text("invalid,header\n", encoding="utf-8")
    monkeypatch.setattr(attendance, "load_employees", lambda directory: [registered_employee()])
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: pytest.fail("camera should remain closed"))
    assert attendance.main(["--attendance-csv", str(path)]) == 2
    assert path.read_text(encoding="utf-8") == "invalid,header\n"
    assert "Invalid attendance CSV header" in capsys.readouterr().err


def test_main_reports_missing_employee_directory_before_opening_camera(monkeypatch, capsys):
    monkeypatch.setattr(attendance, "load_employees", lambda directory: (_ for _ in ()).throw(FileNotFoundError(directory)))
    monkeypatch.setattr(attendance.camera, "open_camera", lambda index: pytest.fail("camera should remain closed"))
    assert attendance.main(["--employees", "missing"]) == 2
    assert "missing" in capsys.readouterr().err
