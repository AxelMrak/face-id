from datetime import datetime, timezone

import numpy as np
import pytest

from facial_recognition.attendance_csv import AttendanceCSV
from facial_recognition.attendance_service import (AttendanceService,
                                                   AttendanceTracker)
from facial_recognition.attendance_session import (AttendanceSession,
                                                   recognize_frame)
from facial_recognition.models import AttendanceStatus, DetectedFace, Employee


def test_recognize_frame_converts_bgr_and_ignores_multiple_faces():
    employee = Employee("E1", "Ana", np.zeros(128))
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    faces = [DetectedFace((1, 20, 20, 1), np.zeros(128))]
    def extract(rgb):
        assert tuple(rgb[0, 0]) == (3, 2, 1)
        return faces
    observation = recognize_frame(frame, [employee], 0.6, extractor=extract)
    assert observation.employee == employee
    faces.append(DetectedFace((1, 20, 20, 1), np.ones(128)))
    assert recognize_frame(frame, [employee], 0.6, extractor=extract).employee is None


def test_session_resets_on_invalid_frame_and_only_confirms_after_save(tmp_path):
    employee = Employee("E1", "Ana", np.zeros(128))
    face = DetectedFace((1, 20, 20, 1), np.zeros(128))
    tracker = AttendanceTracker(AttendanceService(AttendanceCSV(tmp_path / "attendance.csv")))
    session = AttendanceSession([employee], 0.6, tracker)
    session.extractor = lambda rgb: [face]
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    assert session.process(frame, now).attendance is None
    session.extractor = lambda rgb: (_ for _ in ()).throw(ValueError("invalid"))
    assert session.process(frame, now).invalid_encoding
    session.extractor = lambda rgb: [face]
    assert session.process(frame, now).attendance is None
    assert session.process(frame, now).attendance is None
    registered = session.process(frame, now).attendance
    assert registered is not None
    assert registered.status is AttendanceStatus.REGISTERED


def test_session_propagates_csv_failure(tmp_path, monkeypatch):
    employee = Employee("E1", "Ana", np.zeros(128))
    store = AttendanceCSV(tmp_path / "attendance.csv")
    monkeypatch.setattr(store, "append_record", lambda record: (_ for _ in ()).throw(OSError("disk full")))
    session = AttendanceSession([employee], 0.6, AttendanceTracker(AttendanceService(store)))
    session.extractor = lambda rgb: [DetectedFace((1, 20, 20, 1), np.zeros(128))]
    now = datetime.now().astimezone()
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    session.process(frame, now)
    session.process(frame, now)
    with pytest.raises(OSError, match="disk full"):
        session.process(frame, now)
