from typing import cast

import cv2
import numpy as np
import pytest

from facial_recognition import camera


class FakeCapture:
    def __init__(self, opened=True, reads=()):
        self.opened = opened
        self.reads = iter(reads)
        self.settings = []
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, property_id, value):
        self.settings.append((property_id, value))
        return True

    def read(self):
        return next(self.reads)

    def release(self):
        self.released = True


def test_open_camera_requests_resolution(monkeypatch):
    capture = FakeCapture()
    indices = []
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda index: indices.append(index) or capture)

    assert camera.open_camera(2) is capture
    assert indices == [2]
    assert (camera.cv2.CAP_PROP_FRAME_WIDTH, 640) in capture.settings
    assert (camera.cv2.CAP_PROP_FRAME_HEIGHT, 480) in capture.settings


def test_open_camera_releases_unavailable_device(monkeypatch):
    capture = FakeCapture(opened=False)
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda index: capture)
    with pytest.raises(RuntimeError, match="camera"):
        camera.open_camera(0)
    assert capture.released


def test_read_frame_reports_capture_failure():
    with pytest.raises(RuntimeError, match="capture"):
        camera.read_frame(cast(cv2.VideoCapture, FakeCapture(reads=[(False, None)])))


def test_read_frame_returns_image():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert camera.read_frame(cast(cv2.VideoCapture, FakeCapture(reads=[(True, frame)]))) is frame


@pytest.mark.parametrize(("key", "quit_requested"), [(ord("q"), True), (ord("Q"), True), (-1, False)])
def test_show_frame_quits_only_on_q(monkeypatch, key, quit_requested):
    shown = []
    monkeypatch.setattr(camera.cv2, "imshow", lambda title, frame: shown.append((title, frame)))
    monkeypatch.setattr(camera.cv2, "waitKey", lambda delay: key)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert camera.show_frame(frame) is quit_requested
    assert shown == [("Facial Attendance - Live", frame)]


def test_close_camera_releases_even_if_window_cleanup_fails(monkeypatch):
    capture = FakeCapture()
    monkeypatch.setattr(camera.cv2, "destroyAllWindows", lambda: (_ for _ in ()).throw(RuntimeError("window")))
    with pytest.raises(RuntimeError, match="window"):
        camera.close_camera(cast(cv2.VideoCapture, capture))
    assert capture.released
