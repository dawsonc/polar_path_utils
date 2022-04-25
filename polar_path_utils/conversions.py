"""Useful conversion functions"""
import numpy as np


def cartesian_to_polar(pts: np.ndarray):
    """Convert the given points to cartesian.

    args:
        pts: either a (2,) or (N, 2) array of (x, y) points
    returns:
        the points converted into polar coordinates
    """
    one_dimensional = False
    if pts.ndim == 1:
        one_dimensional = True
        pts = pts.reshape(-1, 2)

    # Convert to polar
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    theta = np.arctan2(pts[:, 1], pts[:, 0])

    polar_pts = np.hstack((r.reshape(-1, 1), theta.reshape(-1, 1)))

    if one_dimensional:
        polar_pts = polar_pts.reshape(-1)

    return polar_pts


def polar_to_cartesian(pts: np.ndarray):
    """Convert the given points to polar.

    args:
        pts: either a (2,) or (N, 2) array of (r, theta) points
    returns:
        the points converted into cartesian coordinates
    """
    one_dimensional = False
    if pts.ndim == 1:
        one_dimensional = True
        pts = pts.reshape(-1, 2)

    # Convert to cartesian
    x = pts[:, 0] * np.cos(pts[:, 1])
    y = pts[:, 0] * np.sin(pts[:, 1])

    polar_pts = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

    if one_dimensional:
        polar_pts = polar_pts.reshape(-1)

    return polar_pts
