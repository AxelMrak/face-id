import numpy as np
import pytest

from facial_recognition.engine import find_best_match
from facial_recognition.models import Employee


def employees(*pairs):
    return [Employee(identity, identity, np.array([value, 0.0])) for identity, value in pairs]


def test_selects_nearest_employee_even_when_list_is_unsorted():
    result = find_best_match(np.array([0.0, 0.0]), employees(("far", 0.4), ("near", 0.2)), 0.3)
    assert result.employee_id == "near"
    assert result.distance == pytest.approx(0.2)
    assert result.threshold == 0.3


def test_accepts_distance_equal_to_threshold():
    result = find_best_match(np.array([0.0, 0.0]), employees(("exact", 0.5)), 0.5)
    assert result.employee_id == "exact"
    assert result.distance == pytest.approx(0.5)


def test_unknown_face_preserves_nearest_distance():
    result = find_best_match(np.array([0.0, 0.0]), employees(("far", 0.7), ("near", 0.4)), 0.3)
    assert result.employee_id is None
    assert result.distance == pytest.approx(0.4)


def test_no_employees_has_no_candidate_distance():
    result = find_best_match(np.zeros(2), [])
    assert result.employee_id is None
    assert result.distance is None
    assert result.threshold == 0.6


def test_invalid_threshold_is_rejected_even_with_no_employees():
    with pytest.raises(ValueError, match="threshold"):
        find_best_match(np.zeros(2), [], -1)
