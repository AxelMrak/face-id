import sys
from types import SimpleNamespace

import numpy as np
import pytest

from facial_recognition.models import DetectedFace
from scripts import compare_images


def test_cli_reports_missing_file_without_opening_window(monkeypatch, capsys):
    def load_image_file(path):
        raise FileNotFoundError(path)

    monkeypatch.setitem(sys.modules, "face_recognition", SimpleNamespace(load_image_file=load_image_file))
    monkeypatch.setattr(compare_images.cv2, "imshow", lambda *args: pytest.fail("unexpected window"))

    assert compare_images.main(["--image-a", "missing.jpg", "--image-b", "b.jpg"]) == 2
    assert "missing.jpg" in capsys.readouterr().err


def test_cli_reports_corrupt_file(monkeypatch, capsys):
    def load_image_file(path):
        raise OSError("cannot identify image file")

    monkeypatch.setitem(sys.modules, "face_recognition", SimpleNamespace(load_image_file=load_image_file))
    assert compare_images.main(["--image-a", "corrupt.jpg", "--image-b", "b.jpg"]) == 2
    assert "cannot identify image file" in capsys.readouterr().err


def test_cli_rejects_multiple_faces(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "face_recognition", SimpleNamespace(
        load_image_file=lambda path: np.zeros((20, 20, 3), dtype=np.uint8)
    ))
    monkeypatch.setattr(compare_images, "extract_faces", lambda image: [
        DetectedFace((1, 10, 10, 1), np.zeros(128)),
        DetectedFace((2, 19, 12, 11), np.ones(128)),
    ])
    monkeypatch.setattr(compare_images.cv2, "imshow", lambda *args: pytest.fail("unexpected window"))

    assert compare_images.main(["--image-a", "a.jpg", "--image-b", "b.jpg"]) == 2
    assert "exactly one" in capsys.readouterr().err


def test_cli_reports_no_faces(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "face_recognition", SimpleNamespace(
        load_image_file=lambda path: np.zeros((20, 20, 3), dtype=np.uint8)
    ))
    monkeypatch.setattr(compare_images, "extract_faces", lambda image: [])
    monkeypatch.setattr(compare_images.cv2, "imshow", lambda *args: pytest.fail("unexpected window"))

    assert compare_images.main(["--image-a", "a.jpg", "--image-b", "b.jpg"]) == 2
    assert "no faces" in capsys.readouterr().err


def test_cli_reports_no_match(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "face_recognition", SimpleNamespace(
        load_image_file=lambda path: np.full((20, 20, 3), 0 if path == "a.jpg" else 1, dtype=np.uint8)
    ))
    monkeypatch.setattr(compare_images, "extract_faces", lambda image: [
        DetectedFace((1, 10, 10, 1), np.full(128, float(image[0, 0, 0])))
    ])
    monkeypatch.setattr(compare_images.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(compare_images.cv2, "waitKey", lambda delay: 0)
    monkeypatch.setattr(compare_images.cv2, "destroyAllWindows", lambda: None)

    assert compare_images.main(["--image-a", "a.jpg", "--image-b", "b.jpg"]) == 0
    assert "Result: NO_MATCH" in capsys.readouterr().out


def test_cli_displays_match_with_bounding_boxes(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "face_recognition", SimpleNamespace(
        load_image_file=lambda path: np.zeros((20, 20, 3), dtype=np.uint8)
    ))
    monkeypatch.setattr(compare_images, "extract_faces", lambda image: [
        DetectedFace((1, 10, 10, 1), np.zeros(128))
    ])
    shown = []
    monkeypatch.setattr(compare_images.cv2, "imshow", lambda title, image: shown.append(image.copy()))
    monkeypatch.setattr(compare_images.cv2, "waitKey", lambda delay: 0)
    monkeypatch.setattr(compare_images.cv2, "destroyAllWindows", lambda: None)

    assert compare_images.main(["--image-a", "a.jpg", "--image-b", "b.jpg"]) == 0
    assert "Result: MATCH" in capsys.readouterr().out
    assert len(shown) == 2
    assert shown[0][1, 1].any()  # rectangle drawn over black input
