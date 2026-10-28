"""Exercise the camera-to-recognition-to-CSV path with committed photos, without hardware."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from facial_recognition.attendance_csv import AttendanceCSV
from facial_recognition.attendance_service import (AttendanceService,
                                                   AttendanceTracker)
from facial_recognition.employee_registry import load_employees
from scripts.attendance import annotate_frame

ROOT = Path(__file__).resolve().parents[1]


def test_real_photo_records_once_and_survives_restart(tmp_path):
    employees = load_employees(str(ROOT / "data/employees"))
    frame = cv2.imread(str(ROOT / "data/test-images/alberto-2.jpeg"))
    assert frame is not None
    path = tmp_path / "attendance.csv"
    when = datetime(2026, 10, 28, 8, 5, tzinfo=timezone(timedelta(hours=-3)))
    tracker = AttendanceTracker(AttendanceService(AttendanceCSV(path)))

    for _ in range(3):
        annotate_frame(frame, employees, 0.6, on_match=lambda employee: tracker.observe(employee, when))
    records = AttendanceCSV(path).load_records()
    assert len(records) == 1
    assert records[0].employee_id == "AR001"
    assert records[0].employee_name == "Alberto Fernandez"
    assert records[0].check_in_at == when

    restarted = AttendanceTracker(AttendanceService(AttendanceCSV(path)))
    for _ in range(3):
        annotate_frame(frame, employees, 0.6, on_match=lambda employee: restarted.observe(employee, when))
    assert AttendanceCSV(path).load_records() == records
