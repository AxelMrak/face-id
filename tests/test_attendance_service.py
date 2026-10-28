from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from facial_recognition.attendance_csv import AttendanceCSV
from facial_recognition.attendance_service import (AttendanceService,
                                                   AttendanceTracker)
from facial_recognition.models import AttendanceStatus, Employee

ZONE = timezone(timedelta(hours=-3))
MORNING = datetime(2026, 10, 28, 8, 5, tzinfo=ZONE)
EMPLOYEE = Employee("EMP001", "Ana García", np.zeros(128))


def test_registers_once_per_employee_local_day_across_restarts(tmp_path):
    store = AttendanceCSV(tmp_path / "attendance.csv")
    service = AttendanceService(store)
    first = service.register(EMPLOYEE, MORNING)
    assert first.status is AttendanceStatus.REGISTERED
    assert first.record.employee_id == "EMP001"
    assert first.record.employee_name == "Ana García"
    assert first.record.check_in_at == MORNING

    after_restart = AttendanceService(AttendanceCSV(store.path))
    duplicate = after_restart.register(EMPLOYEE, MORNING + timedelta(hours=3))
    assert duplicate.status is AttendanceStatus.ALREADY_REGISTERED
    assert duplicate.record == first.record
    assert store.load_records() == [first.record]

    next_day = after_restart.register(EMPLOYEE, MORNING + timedelta(days=1))
    assert next_day.status is AttendanceStatus.REGISTERED
    assert len(store.load_records()) == 2


def test_different_employees_with_same_name_are_not_duplicates(tmp_path):
    service = AttendanceService(AttendanceCSV(tmp_path / "attendance.csv"))
    other = Employee("EMP002", EMPLOYEE.name, np.ones(128))
    assert service.register(EMPLOYEE, MORNING).status is AttendanceStatus.REGISTERED
    assert service.register(other, MORNING).status is AttendanceStatus.REGISTERED


def test_write_failure_does_not_confirm_or_poison_in_memory_index(tmp_path, monkeypatch):
    store = AttendanceCSV(tmp_path / "attendance.csv")
    service = AttendanceService(store)
    original_append = store.append_record
    monkeypatch.setattr(store, "append_record", lambda record: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        service.register(EMPLOYEE, MORNING)
    monkeypatch.setattr(store, "append_record", original_append)

    assert service.register(EMPLOYEE, MORNING).status is AttendanceStatus.REGISTERED
    assert len(store.load_records()) == 1


def test_tracker_requires_three_consecutive_identifications_and_resets_on_unknown(tmp_path):
    store = AttendanceCSV(tmp_path / "attendance.csv")
    tracker = AttendanceTracker(AttendanceService(store))
    assert tracker.observe(EMPLOYEE, MORNING) is None
    assert tracker.observe(None, MORNING) is None
    assert tracker.observe(EMPLOYEE, MORNING) is None
    assert tracker.observe(EMPLOYEE, MORNING) is None
    result = tracker.observe(EMPLOYEE, MORNING)
    assert result is not None and result.status is AttendanceStatus.REGISTERED
    assert tracker.observe(EMPLOYEE, MORNING) == result
    assert len(store.load_records()) == 1

    assert tracker.observe(None, MORNING) is None
    assert tracker.observe(EMPLOYEE, MORNING) is None
    assert tracker.observe(EMPLOYEE, MORNING) is None
    repeated = tracker.observe(EMPLOYEE, MORNING)
    assert repeated is not None and repeated.status is AttendanceStatus.ALREADY_REGISTERED
    assert len(store.load_records()) == 1


def test_tracker_resets_on_other_employee_and_new_day(tmp_path):
    store = AttendanceCSV(tmp_path / "attendance.csv")
    tracker = AttendanceTracker(AttendanceService(store))
    other = Employee("EMP002", "Juan", np.ones(128))
    assert tracker.observe(EMPLOYEE, MORNING) is None
    assert tracker.observe(other, MORNING) is None
    assert tracker.observe(EMPLOYEE, MORNING) is None
    assert tracker.observe(EMPLOYEE, MORNING) is None
    registered = tracker.observe(EMPLOYEE, MORNING)
    assert registered is not None and registered.status is AttendanceStatus.REGISTERED
    next_day = MORNING + timedelta(days=1)
    assert tracker.observe(EMPLOYEE, next_day) is None
    assert tracker.observe(EMPLOYEE, next_day) is None
    next_registered = tracker.observe(EMPLOYEE, next_day)
    assert next_registered is not None and next_registered.status is AttendanceStatus.REGISTERED
    assert len(store.load_records()) == 2


def test_rejects_timestamp_without_timezone(tmp_path):
    service = AttendanceService(AttendanceCSV(tmp_path / "attendance.csv"))
    with pytest.raises(ValueError, match="timezone"):
        service.register(EMPLOYEE, datetime(2026, 10, 28, 8, 5))
    assert not (tmp_path / "attendance.csv").exists()
