import numpy as np
import matplotlib.pyplot as plt

from polar_path_utils.lines import (
    plan_line,
)
from polar_path_utils.plotting_utils import fix_polar_points_for_plotting


def test_plan_line():
    # Plan a line that is nearly singular
    duration = 5.0
    timestep = 0.01
    line_start_pt_polar = np.array([1.0, 0.0])
    line_end_pt_polar = np.array([1.0, np.pi - 0.01])
    radial_speed_limit = 1.0
    angular_speed_limit = np.pi

    t, pts, velocities = plan_line(
        line_start_pt_polar,
        line_end_pt_polar,
        duration,
        timestep,
        radial_speed_limit,
        angular_speed_limit,
    )

    # Times should be correct
    assert np.isclose(t.min(), 0.0)
    assert t.max() <= duration
    assert np.allclose(np.diff(t).min(), timestep)
    assert np.allclose(np.diff(t).max(), timestep)

    # Make sure the waypoints are consistent
    assert pts.shape == (t.shape[0], 2)
    assert velocities.shape == (t.shape[0], 2)

    # Make sure the velocity constraints are respected
    tolerance = 1e-3
    assert np.abs(velocities[:, 0]).max() <= radial_speed_limit + tolerance
    assert np.abs(velocities[:, 1]).max() <= angular_speed_limit + tolerance


def plan_line_no_singularity(duration: float, timestep: float):
    line_start_pt_polar = np.array([1.0, 0.0])
    line_end_pt_polar = np.array([1.0, np.pi - 1])

    return plan_line(line_start_pt_polar, line_end_pt_polar, duration, timestep)


def plan_line_close_to_singularity(
    duration: float, timestep: float, how_close: float = 0.01
):
    line_start_pt_polar = np.array([1.0, 0.0])
    line_end_pt_polar = np.array([1.0, np.pi - how_close])

    return plan_line(line_start_pt_polar, line_end_pt_polar, duration, timestep)


def plan_line_singular(duration: float, timestep: float):
    line_start_pt_polar = np.array([1.0, 0.0])
    line_end_pt_polar = np.array([1.0, np.pi])

    return plan_line(line_start_pt_polar, line_end_pt_polar, duration, timestep)


def plot_lines():
    duration = 5.0
    timestep = 0.01

    # Plan an arc and plot
    _, line_pts, _ = plan_line_no_singularity(duration, timestep)
    line_pts = fix_polar_points_for_plotting(line_pts)
    plt.polar(line_pts[:, 1], line_pts[:, 0], label="Non-singular Line")

    for close_to_singularity in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
        _, line_pts, _ = plan_line_close_to_singularity(
            duration, timestep, close_to_singularity
        )
        line_pts = fix_polar_points_for_plotting(line_pts)
        plt.polar(line_pts[:, 1], line_pts[:, 0], label="Near-singular Line")

    _, line_pts, _ = plan_line_singular(duration, timestep)
    line_pts = fix_polar_points_for_plotting(line_pts)
    plt.polar(line_pts[:, 1], line_pts[:, 0], label="Singular Line")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_lines()
