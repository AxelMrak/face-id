"""Local CSV persistence for attendance records."""

import csv
from datetime import datetime
from pathlib import Path

from facial_recognition.models import AttendanceRecord

FIELDS = ("employee_id", "employee_name", "check_in_at")


class AttendanceCSV:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load_records(self) -> list[AttendanceRecord]:
        if not self.path.exists():
            return []
        with self.path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                return []
            if tuple(reader.fieldnames) != FIELDS:
                raise ValueError(f"Invalid attendance CSV header: {self.path}")
            records = []
            for row in reader:
                if not all(row.get(field) for field in FIELDS):
                    raise ValueError(f"Invalid attendance CSV row: {self.path}")
                check_in_at = datetime.fromisoformat(row["check_in_at"])
                if check_in_at.tzinfo is None or check_in_at.utcoffset() is None:
                    raise ValueError(f"Attendance timestamp must have a timezone: {self.path}")
                records.append(AttendanceRecord(row["employee_id"], row["employee_name"], check_in_at))
            return records

    def append_record(self, record: AttendanceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(FIELDS)
            writer.writerow((record.employee_id, record.employee_name, record.check_in_at.isoformat()))
