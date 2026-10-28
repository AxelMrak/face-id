"""Identify employees from a local webcam and record daily check-ins."""

import argparse
import sys
from collections.abc import Callable
from datetime import datetime

import cv2
import numpy as np

from facial_recognition import camera
from facial_recognition.attendance_csv import AttendanceCSV
from facial_recognition.attendance_service import (AttendanceService,
                                                   AttendanceTracker)
from facial_recognition.employee_registry import load_employees
from facial_recognition.engine import extract_faces, find_best_match
from facial_recognition.models import (AttendanceResult, AttendanceStatus,
                                       Employee, RecognizedFace)


def annotate_frame(
    frame_bgr: np.ndarray,
    employees: list[Employee],
    threshold: float,
    on_match: Callable[[Employee | None], AttendanceResult | None] | None = None,
) -> np.ndarray:
    """Detect in RGB, then annotate the original BGR frame for display."""
    try:
        faces = extract_faces(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    except ValueError as exc:
        print(f"Skipping frame with invalid encoding: {exc}", file=sys.stderr)
        faces = []
    annotated = frame_bgr.copy()
    matched_employee = None
    if len(faces) > 1:
        cv2.putText(annotated, "MULTIPLE_FACES", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for face in faces:
            top, right, bottom, left = face.location
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 255), 2)
    elif faces:
        face = faces[0]
        result = find_best_match(face.encoding, employees, threshold=threshold)
        recognized = RecognizedFace(face.location, result.employee_id, result.distance)
        matched_employee = next((person for person in employees if person.id == recognized.employee_id), None)
        label = matched_employee.name if matched_employee is not None else "UNKNOWN"
        color = (0, 255, 0) if matched_employee is not None else (0, 0, 255)
        top, right, bottom, left = recognized.location
        cv2.putText(annotated, label, (left, max(top - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)

    attendance = on_match(matched_employee) if on_match is not None else None
    if attendance is not None:
        status = "Attendance registered" if attendance.status is AttendanceStatus.REGISTERED else "Already registered today"
        cv2.putText(annotated, status, (10, max(15, annotated.shape[0] - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        check_in = f"Check-in: {attendance.record.check_in_at:%H:%M:%S}"
        cv2.putText(annotated, check_in, (10, max(35, annotated.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Maximum face distance (default: 0.6)")
    parser.add_argument("--employees", default="data/employees", help="Directory of registered employee photos")
    parser.add_argument("--attendance-csv", default="data/attendance.csv", help="Check-in CSV path")
    args = parser.parse_args(argv)
    if not np.isfinite(args.threshold) or args.threshold < 0:
        print("Threshold must be finite and non-negative", file=sys.stderr)
        return 2

    try:
        employees = load_employees(args.employees)
        tracker = AttendanceTracker(AttendanceService(AttendanceCSV(args.attendance_csv)))

        def on_match(employee: Employee | None) -> AttendanceResult | None:
            return tracker.observe(employee, datetime.now().astimezone())

        capture = camera.open_camera(args.camera)
        try:
            capture_failures = 0
            while True:
                try:
                    frame = camera.read_frame(capture)
                except RuntimeError as exc:
                    tracker.observe(None, datetime.now().astimezone())
                    capture_failures += 1
                    if capture_failures >= 3:
                        raise
                    print(f"Camera capture failed ({capture_failures}/3); retrying: {exc}", file=sys.stderr)
                    continue
                capture_failures = 0
                if camera.show_frame(annotate_frame(frame, employees, args.threshold, on_match=on_match)):
                    return 0
        finally:
            camera.close_camera(capture)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"Live recognition failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
