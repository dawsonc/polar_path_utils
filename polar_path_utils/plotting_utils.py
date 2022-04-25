import numpy as np


def fix_polar_points_for_plotting(polar_pts):
    """
    Fix the given polar points so that they can be plotted using matplotlib's polar
    function, which expects the radius to always be positive.

    args:
        polar_pts: an (n_pts, 2) array of [r, theta] polar points
    """
    # Make a copy
    fixed_pts = np.array(polar_pts)

    # Wherever the radius is negative, add pi to the angle instead
    fixed_pts[:, 1] = np.where(
        fixed_pts[:, 0] < 0, fixed_pts[:, 1] + np.pi, fixed_pts[:, 1]
    )
    fixed_pts[:, 0] = np.where(fixed_pts[:, 0] < 0, -fixed_pts[:, 0], fixed_pts[:, 0])

    return fixed_pts
