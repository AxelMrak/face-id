"""OpenCV camera capture and window lifecycle."""

import cv2
import numpy as np


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera and request a 640 × 480 stream."""
    if index < 0:
        raise ValueError("camera index must be non-negative")
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera {index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return capture


def read_frame(capture: cv2.VideoCapture) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError("Camera capture failed")
    return frame


def show_frame(frame: np.ndarray) -> bool:
    """Show a frame and return True when Q is pressed in the window."""
    cv2.imshow("Facial Attendance - Live", frame)
    return cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"))


def close_camera(capture: cv2.VideoCapture) -> None:
    try:
        capture.release()
    finally:
        cv2.destroyAllWindows()
