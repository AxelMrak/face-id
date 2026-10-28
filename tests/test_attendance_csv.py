import csv
from datetime import datetime, timedelta, timezone

import pytest

from facial_recognition.attendance_csv import AttendanceCSV
from facial_recognition.models import AttendanceRecord

ZONE = timezone(timedelta(hours=-3))
TIME = datetime(2026, 10, 28, 8, 5, 14, tzinfo=ZONE)


def test_creates_csv_with_header_and_timezone_aware_row(tmp_path):
    store = AttendanceCSV(tmp_path / "data" / "attendance.csv")
    assert store.load_records() == []
    record = AttendanceRecord("EMP001", "Ana García", TIME)
    store.append_record(record)

    with store.path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows == [
        ["employee_id", "employee_name", "check_in_at"],
        ["EMP001", "Ana García", "2026-10-28T08:05:14-03:00"],
    ]
    assert store.load_records() == [record]


def test_appends_without_overwriting_history_or_repeating_header(tmp_path):
    store = AttendanceCSV(tmp_path / "attendance.csv")
    first = AttendanceRecord("EMP001", "Ana", TIME)
    second = AttendanceRecord("EMP002", "Juan", TIME + timedelta(minutes=5))
    store.append_record(first)
    store.append_record(second)

    assert AttendanceCSV(store.path).load_records() == [first, second]
    assert store.path.read_text(encoding="utf-8").count("employee_id,employee_name,check_in_at") == 1


def test_rejects_corrupt_existing_csv_instead_of_discarding_history(tmp_path):
    path = tmp_path / "attendance.csv"
    path.write_text("wrong,header\nEMP001,Ana\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        AttendanceCSV(path).load_records()
    assert path.read_text(encoding="utf-8") == "wrong,header\nEMP001,Ana\n"


def test_rejects_extra_csv_columns_without_altering_file(tmp_path):
    path = tmp_path / "attendance.csv"
    content = "employee_id,employee_name,check_in_at\nEMP001,Ana,2026-10-28T08:05:14-03:00,extra\n"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="row"):
        AttendanceCSV(path).load_records()
    assert path.read_text(encoding="utf-8") == content


def test_rejects_naive_timestamp_when_loading(tmp_path):
    path = tmp_path / "attendance.csv"
    path.write_text("employee_id,employee_name,check_in_at\nEMP001,Ana,2026-10-28T08:05:14\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timezone"):
        AttendanceCSV(path).load_records()
