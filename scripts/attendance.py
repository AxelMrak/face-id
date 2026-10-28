"""Identify employees from a local webcam; no attendance is stored."""

import argparse
import sys

import cv2
import numpy as np

from facial_recognition import camera
from facial_recognition.employee_registry import load_employees
from facial_recognition.engine import extract_faces, find_best_match
from facial_recognition.models import Employee, RecognizedFace


def annotate_frame(frame_bgr: np.ndarray, employees: list[Employee], threshold: float) -> np.ndarray:
    """Detect in RGB, then annotate the original BGR frame for display."""
    faces = extract_faces(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    annotated = frame_bgr.copy()
    if len(faces) > 1:
        cv2.putText(annotated, "MULTIPLE_FACES", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for face in faces:
            top, right, bottom, left = face.location
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 255), 2)
    elif faces:
        face = faces[0]
        result = find_best_match(face.encoding, employees, threshold=threshold)
        recognized = RecognizedFace(face.location, result.employee_id, result.distance)
        label = next((person.name for person in employees if person.id == recognized.employee_id), "UNKNOWN")
        color = (0, 255, 0) if recognized.employee_id is not None else (0, 0, 255)
        top, right, bottom, left = recognized.location
        cv2.putText(annotated, label, (left, max(top - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
    return annotated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Maximum face distance (default: 0.6)")
    parser.add_argument("--employees", default="data/employees", help="Directory of registered employee photos")
    args = parser.parse_args(argv)
    if not np.isfinite(args.threshold) or args.threshold < 0:
        print("Threshold must be finite and non-negative", file=sys.stderr)
        return 2

    try:
        employees = load_employees(args.employees)
        capture = camera.open_camera(args.camera)
        try:
            while True:
                frame = camera.read_frame(capture)
                if camera.show_frame(annotate_frame(frame, employees, args.threshold)):
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
