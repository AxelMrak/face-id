"""Offscreen smoke tests for the desktop workflow (no camera required)."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from facial_recognition.ui import main_window


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
        assert window.pages.count() == 3
        window.nav_compare.click()
        assert window.pages.currentIndex() == 1
        window.path_a.setText("a.jpg")
        window.path_b.setText("b.jpg")
        monkeypatch.setattr(main_window, "compare_files", lambda *args: (True, 0.32))
        window.compare_button.click()
        assert not window.compare_button.isEnabled()
        wait_for(app, lambda: window.compare_button.isEnabled())
        assert "0.3200" in window.compare_result.text()
        window.nav_tests.click()
        assert window.pages.currentIndex() == 2
    finally:
        window.close()


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
