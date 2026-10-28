"""Load local employee portraits and keep their encodings in memory."""

import warnings
from pathlib import Path

from facial_recognition.engine import extract_faces
from facial_recognition.models import Employee

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_employees(directory: str) -> list[Employee]:
    """Load one portrait per ID; skip unusable photos and reject duplicate IDs."""
    import face_recognition

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Employee directory not found: {directory}")

    employees = []
    seen_ids = set()
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        employee_id, separator, raw_name = path.stem.partition("__")
        name_parts = raw_name.split("_")
        if not separator or not employee_id.isalnum() or not all(name_parts):
            warnings.warn(f"{path.name}: invalid employee filename", RuntimeWarning, stacklevel=2)
            continue
        if employee_id in seen_ids:
            raise ValueError(f"Duplicate employee ID: {employee_id}")
        seen_ids.add(employee_id)
        try:
            faces = extract_faces(face_recognition.load_image_file(path))
        except (OSError, ValueError) as exc:
            warnings.warn(f"{path.name}: {exc}", RuntimeWarning, stacklevel=2)
            continue
        if len(faces) != 1:
            warnings.warn(f"{path.name}: exactly one face is required", RuntimeWarning, stacklevel=2)
            continue
        employees.append(Employee(employee_id, " ".join(name_parts), faces[0].encoding))

    if not employees:
        raise ValueError(f"Employee directory has no valid employees: {directory}")
    return employees
