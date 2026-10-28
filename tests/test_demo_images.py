"""End-to-end comparison tests using the local demo photographs, when available."""

from pathlib import Path

import pytest

from scripts import compare_images

DEMO_IMAGES = Path(__file__).parent.parent / "demo-images"


@pytest.mark.parametrize(
    ("image_a", "image_b", "expected"),
    [
        ("trump-1.jpeg", "trump-2.jpeg", "MATCH"),
        ("biden-1.jpeg", "biden-2.jpg", "MATCH"),
        ("trump-1.jpeg", "biden-1.jpeg", "NO_MATCH"),
    ],
    ids=["same-person-trump", "same-person-biden", "different-people"],
)
def test_demo_image_comparison(monkeypatch, capsys, image_a, image_b, expected):
    if not (DEMO_IMAGES / image_a).is_file() or not (DEMO_IMAGES / image_b).is_file():
        pytest.skip("local demo-images are not available")

    monkeypatch.setattr(compare_images.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(compare_images.cv2, "waitKey", lambda delay: 0)
    monkeypatch.setattr(compare_images.cv2, "destroyAllWindows", lambda: None)

    exit_code = compare_images.main([
        "--image-a", str(DEMO_IMAGES / image_a),
        "--image-b", str(DEMO_IMAGES / image_b),
        "--threshold", "0.6",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"Result: {expected}" in output
    assert "Distance: " in output
