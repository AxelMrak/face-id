"""Decide whether to record an employee's first check-in of the day."""

from datetime import datetime

from facial_recognition.attendance_csv import AttendanceCSV
from facial_recognition.models import (AttendanceRecord, AttendanceResult,
                                       AttendanceStatus, Employee)


class AttendanceService:
    def __init__(self, store: AttendanceCSV):
        self.store = store
        self.records = {(record.employee_id, record.check_in_at.date()): record for record in store.load_records()}

    def register(self, employee: Employee, check_in_at: datetime) -> AttendanceResult:
        if check_in_at.tzinfo is None or check_in_at.utcoffset() is None:
            raise ValueError("Check-in timestamp must have a timezone")
        key = (employee.id, check_in_at.date())
        if key in self.records:
            return AttendanceResult(AttendanceStatus.ALREADY_REGISTERED, self.records[key])
        record = AttendanceRecord(employee.id, employee.name, check_in_at)
        self.store.append_record(record)
        self.records[key] = record
        return AttendanceResult(AttendanceStatus.REGISTERED, record)


class AttendanceTracker:
    """Require three consecutive frames with the same employee before registering."""

    def __init__(self, service: AttendanceService):
        self.service = service
        self.employee_id: str | None = None
        self.date = None
        self.count = 0
        self.result: AttendanceResult | None = None

    def observe(self, employee: Employee | None, check_in_at: datetime) -> AttendanceResult | None:
        employee_id = employee.id if employee is not None else None
        if employee_id is None or employee_id != self.employee_id or check_in_at.date() != self.date:
            self.employee_id = employee_id
            self.date = check_in_at.date()
            self.count = 0
            self.result = None
        if employee is None:
            return None
        self.count += 1
        if self.count == 3:
            self.result = self.service.register(employee, check_in_at)
        return self.result
