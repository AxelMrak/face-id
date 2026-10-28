"""Offscreen smoke tests for the desktop workflow (no camera required)."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QProcess
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QScrollArea

from facial_recognition.ui import main_window
from facial_recognition.ui.workers import ComparisonView


def wait_for(app, predicate, timeout=5):
    until = time.monotonic() + timeout
    while not predicate() and time.monotonic() < until:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


def test_navigation_and_async_comparison(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    try:
        assert window.pages.count() == 4
        window.nav_compare.click()
        assert window.pages.currentIndex() == 1
        window.path_a.setText("a.jpg")
        window.path_b.setText("b.jpg")
        image = QImage(32, 32, QImage.Format.Format_RGB888)
        image.fill("white")
        monkeypatch.setattr(main_window, "compare_files", lambda *args: ComparisonView(True, 0.32, image, image))
        window.compare_button.click()
        assert not window.compare_button.isEnabled()
        wait_for(app, lambda: window.compare_button.isEnabled())
        assert "0.3200" in window.compare_details.text()
        assert window.compare_feedback.property("tone") == "success"
        assert window.photo_a.pixmap() is not None
        assert window.photo_b.pixmap() is not None
        assert "0.60" in window.compare_details.text()
        assert "a.jpg" in window.photo_a_name.text()
        assert "b.jpg" in window.photo_b_name.text()
        window.nav_tests.click()
        assert window.pages.currentIndex() == 2
        window.nav_records.click()
        assert window.pages.currentIndex() == 3
    finally:
        window.close()


def test_non_match_has_visible_warning_and_controls_are_readable(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    try:
        window.show()
        app.processEvents()
        assert window.employees_path.height() >= 40
        assert window.camera_index.height() >= 40
        assert window.csv_path.width() > 260
        window.nav_compare.click()
        window.path_a.setText("a.jpg")
        window.path_b.setText("b.jpg")
        image = QImage(32, 32, QImage.Format.Format_RGB888)
        image.fill("white")
        monkeypatch.setattr(main_window, "compare_files", lambda *args: ComparisonView(False, 0.78, image, image))
        window.compare_button.click()
        wait_for(app, lambda: window.compare_button.isEnabled())
        assert "Sin coincidencia" in window.compare_result.text()
        assert window.compare_icon.text() == "!"
        assert window.compare_feedback.property("tone") == "warning"
        app.processEvents()
        image = window.compare_feedback.grab().toImage()
        assert image.pixelColor(image.width() - 12, image.height() // 2).name() == "#fff5ea"
        window._camera_feedback("Rostro desconocido", "warning")
        assert window.camera_feedback.property("tone") == "warning"
        window._tests_feedback("Pruebas fallidas", "danger")
        assert window.tests_feedback.property("tone") == "danger"
    finally:
        window.close()


def test_attendance_csv_page_loads_records_and_reports_invalid_csv(tmp_path):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "attendance.csv"
    path.write_text("employee_id,employee_name,check_in_at\nEMP001,Ana Garcia,2026-10-28T08:05:00-03:00\n", encoding="utf-8")
    window = main_window.MainWindow()
    try:
        window.csv_path.setText(str(path))
        window.nav_records.click()
        wait_for(app, lambda: window.records_table.rowCount() == 1)
        name = window.records_table.item(0, 1)
        timestamp = window.records_table.item(0, 2)
        assert name is not None and name.text() == "Ana Garcia"
        assert timestamp is not None and "-03:00" in timestamp.text()
        path.write_text("invalid\n", encoding="utf-8")
        window.refresh_records()
        wait_for(app, lambda: window.records_feedback.property("tone") == "danger")
        assert window.records_table.rowCount() == 0
    finally:
        window.close()


def test_comparison_layout_preserves_inputs_and_visible_result():
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    try:
        window.show()
        window.nav_compare.click()
        app.processEvents()
        scroll = window.pages.currentWidget()
        assert isinstance(scroll, QScrollArea)
        assert window.compare_feedback.mapTo(scroll, window.compare_feedback.rect().topLeft()).y() < scroll.height()
        window.resize(900, 700)
        app.processEvents()
        assert window.path_a.width() >= 260
        image = QImage(32, 32, QImage.Format.Format_RGB888)
        image.fill("white")
        window._comparison_done(ComparisonView(False, 0.78, image, image))
        app.processEvents()
        assert scroll.verticalScrollBar().value() > 0
    finally:
        window.close()


def test_cancelled_tests_have_neutral_feedback():
    app = QApplication.instance() or QApplication([])
    window = main_window.MainWindow()
    try:
        window._tests_cancelled = True
        window._tests_finished(-15, QProcess.ExitStatus.CrashExit)
        app.processEvents()
        assert window.tests_status.text() == "Pruebas canceladas"
        assert window.tests_feedback.property("tone") == "neutral"
    finally:
        window.close()


def test_comparison_previews_mark_detected_faces(monkeypatch):
    import face_recognition
    import numpy as np

    from facial_recognition.models import DetectedFace
    from facial_recognition.ui import workers

    photo = np.zeros((60, 60, 3), dtype=np.uint8)
    monkeypatch.setattr(face_recognition, "load_image_file", lambda path: photo)
    monkeypatch.setattr(workers, "extract_faces", lambda image: [DetectedFace((10, 40, 40, 10), np.zeros(128))])
    result = workers.compare_files("a.jpg", "b.jpg", 0.6)
    assert result.is_match
    assert result.image_a.size().width() == 60
    assert result.image_a.pixelColor(10, 10).name() == "#007aff"
    assert result.image_b.pixelColor(10, 10).name() == "#007aff"


def test_camera_worker_stops_after_request_without_event_loop(monkeypatch, tmp_path):
    import numpy as np

    from facial_recognition.models import Employee
    from facial_recognition.ui import workers
    from facial_recognition.ui.workers import CameraWorker

    app = QApplication.instance() or QApplication([])
    released = []
    class Capture:
        def release(self):
            released.append(True)
    monkeypatch.setattr(workers, "load_employees", lambda path: [Employee("E1", "Ana", np.zeros(128))])
    opened = []
    def open_camera(index):
        opened.append(True)
        return Capture()
    monkeypatch.setattr(workers.camera, "open_camera", open_camera)
    def read(capture):
        time.sleep(0.01)
        return np.zeros((40, 40, 3), dtype=np.uint8)
    monkeypatch.setattr(workers.camera, "read_frame", read)
    monkeypatch.setattr(workers, "extract_faces", lambda rgb: [])
    worker = CameraWorker(0, 0.6, "employees", str(tmp_path / "attendance.csv"))
    worker.start()
    try:
        wait_for(app, lambda: opened)
        worker.request_stop()
        assert worker.wait(3000)
        assert released == [True]
    finally:
        worker.request_stop()
        worker.wait(3000)
