"""Background work for the desktop interface; Qt widgets remain on the UI thread."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from threading import Event

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from facial_recognition import camera
from facial_recognition.attendance_csv import AttendanceCSV
from facial_recognition.attendance_service import (AttendanceService,
                                                   AttendanceTracker)
from facial_recognition.attendance_session import AttendanceSession
from facial_recognition.employee_registry import load_employees
from facial_recognition.engine import compare_encodings, extract_faces
from facial_recognition.models import AttendanceStatus


@dataclass(frozen=True)
class ComparisonView:
    is_match: bool
    distance: float
    image_a: QImage
    image_b: QImage


def compare_files(path_a: str, path_b: str, threshold: float) -> ComparisonView:
    import face_recognition

    faces = []
    previews = []
    for path in (path_a, path_b):
        image = face_recognition.load_image_file(path)
        detected = extract_faces(image)
        if len(detected) != 1:
            raise ValueError(f"{path}: exactly one face is required (found {len(detected)})")
        face = detected[0]
        faces.append(face)
        annotated = image.copy()
        top, right, bottom, left = face.location
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 122, 255), 2)
        height, width, _ = annotated.shape
        previews.append(QImage(annotated.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy())
    result = compare_encodings(faces[0].encoding, faces[1].encoding, threshold)
    return ComparisonView(result.is_match, result.distance, previews[0], previews[1])


class JobThread(QThread):
    result = Signal(object)
    error = Signal(str)

    def __init__(self, job: Callable[[], object], parent=None):
        super().__init__(parent)
        self.job = job

    def run(self) -> None:
        try:
            self.result.emit(self.job())
        except Exception as exc:
            self.error.emit(str(exc))


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    status = Signal(str, str)
    error = Signal(str)

    def __init__(self, index: int, threshold: float, employees_dir: str, csv_path: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.threshold = threshold
        self.employees_dir = employees_dir
        self.csv_path = csv_path
        self._stop = Event()

    def request_stop(self) -> None:
        """Safe to call directly from the UI while run() occupies the worker thread."""
        self._stop.set()

    def run(self) -> None:
        capture = None
        try:
            employees = load_employees(self.employees_dir)
            if self._stop.is_set():
                return
            session = AttendanceSession(
                employees, self.threshold, AttendanceTracker(AttendanceService(AttendanceCSV(self.csv_path)))
            )
            session.extractor = extract_faces
            if self._stop.is_set():
                return
            capture = camera.open_camera(self.index)
            self.status.emit("Cámara activa · buscando un rostro", "info")
            failures = 0
            previous_status = ("", "")
            while not self._stop.is_set():
                try:
                    frame = camera.read_frame(capture)
                except RuntimeError:
                    session.reset(datetime.now().astimezone())
                    failures += 1
                    if failures >= 3:
                        raise
                    continue
                failures = 0
                observation = session.process(frame, datetime.now().astimezone())
                if len(observation.faces) > 1:
                    message, tone, color = "Varios rostros · no se registra", "warning", (0, 160, 240)
                elif observation.invalid_encoding:
                    message, tone, color = "No se pudo leer el rostro · reintentando", "warning", (0, 160, 240)
                elif observation.employee is None:
                    message = "Rostro no reconocido" if observation.faces else "Buscando un rostro"
                    tone, color = ("warning" if observation.faces else "info"), (0, 140, 255)
                else:
                    message, tone, color = f"{observation.employee.name} · confirmando identidad", "info", (255, 122, 0)
                if observation.attendance:
                    record = observation.attendance.record
                    registered = observation.attendance.status is AttendanceStatus.REGISTERED
                    label = "Ingreso registrado" if registered else "Ya registrado hoy"
                    message = f"{label} · {record.employee_name} · {record.check_in_at:%H:%M:%S}"
                    tone = "success" if registered else "warning"
                    color = (56, 164, 66) if registered else (0, 160, 240)
                if (message, tone) != previous_status:
                    self.status.emit(message, tone)
                    previous_status = (message, tone)
                for face in observation.faces:
                    top, right, bottom, left = face.location
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = rgb.shape
                self.frame_ready.emit(QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy())
        except Exception as exc:
            if not self._stop.is_set():
                self.error.emit(str(exc))
        finally:
            if capture is not None:
                capture.release()
