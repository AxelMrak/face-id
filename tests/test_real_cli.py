"""Smoke test the installed library without using identifiable photographs."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(importlib.util.find_spec("face_recognition") is None, reason="optional library not installed")
def test_real_cli_rejects_image_without_faces():
    image = Path(__file__).parent / "fixtures" / "no_face.pbm"
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.compare_images", "--image-a", str(image), "--image-b", str(image)],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "no faces" in completed.stderr
