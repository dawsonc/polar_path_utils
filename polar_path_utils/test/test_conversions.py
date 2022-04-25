import numpy as np

from polar_path_utils.conversions import (
    cartesian_to_polar,
    polar_to_cartesian,
)


def test_cartesian_to_polar():
    # Test converting a single point
    test_point_1d = np.array([1.0, 1.0])
    polar = cartesian_to_polar(test_point_1d)
    assert polar.shape == test_point_1d.shape
    assert np.allclose(polar, [np.sqrt(2), np.pi / 4])

    # Test converting multiple points
    test_points = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    expected_polar_pts = np.array(
        [
            [1.0, 0.0],
            [1.0, np.pi],
            [np.sqrt(2), -3 * np.pi / 4],
        ]
    )
    assert np.allclose(cartesian_to_polar(test_points), expected_polar_pts)


def test_polar_to_cartesian():
    # Test converting a single point
    test_point_1d = np.array([np.sqrt(2), np.pi / 4])
    polar = polar_to_cartesian(test_point_1d)
    assert polar.shape == test_point_1d.shape
    assert np.allclose(polar, [1.0, 1.0])

    # Test converting multiple points
    test_points = np.array(
        [
            [1.0, 0.0],
            [1.0, np.pi],
            [np.sqrt(2), -3 * np.pi / 4],
        ]
    )
    expected_cartesian_pts = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    assert np.allclose(polar_to_cartesian(test_points), expected_cartesian_pts)
