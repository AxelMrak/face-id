"""Compare exactly one face per image and display detected bounding boxes."""

import argparse
import sys

import cv2

from facial_recognition.engine import compare_encodings, extract_faces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-a", required=True, help="Reference image")
    parser.add_argument("--image-b", required=True, help="Image to compare")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args(argv)

    import face_recognition

    images = []
    faces_by_image = []
    for label, path in (("A", args.image_a), ("B", args.image_b)):
        try:
            image = face_recognition.load_image_file(path)
            faces = extract_faces(image)
        except (OSError, ValueError) as exc:
            print(f"Image {label} ({path}): {exc}", file=sys.stderr)
            return 2
        print(f"Image {label}: {len(faces)} face(s) detected")
        if len(faces) != 1:
            reason = "no faces detected" if not faces else "exactly one face is required"
            print(f"Image {label} ({path}): {reason}", file=sys.stderr)
            return 2
        images.append(image)
        faces_by_image.append(faces[0])

    try:
        result = compare_encodings(
            faces_by_image[0].encoding, faces_by_image[1].encoding, args.threshold
        )
    except ValueError as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2

    print(f"Distance: {result.distance:.4f}")
    print(f"Threshold: {result.threshold:.2f}")
    print(f"Result: {'MATCH' if result.is_match else 'NO_MATCH'}")
    try:
        for label, image, face in zip("AB", images, faces_by_image, strict=True):
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            top, right, bottom, left = face.location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow(f"Image {label}", frame)
        cv2.waitKey(0)
    except cv2.error as exc:
        print(f"Could not display images: {exc}", file=sys.stderr)
        return 2
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
