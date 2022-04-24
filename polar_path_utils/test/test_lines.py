import numpy as np
import matplotlib.pyplot as plt

from polar_path_utils.lines import (
    plan_line,
    danger_line,
)
from polar_path_utils.plotting_utils import fix_polar_points_for_plotting


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


def plan_danger_line(
    duration: float, timestep: float, how_close: float = 0.01
):
    line_start_pt_polar = np.array([1.0, 0.0])
    line_end_pt_polar = np.array([1.0, np.pi - how_close])

    return danger_line(line_start_pt_polar, line_end_pt_polar, duration, timestep)


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


def plot_line_and_velocity():
    duration = 5.0
    timestep = 0.01

    t, line_pts, line_v = plan_danger_line(duration, timestep, 0.05)
    line_pts = fix_polar_points_for_plotting(line_pts)
    plt.subplot(221, projection="polar")
    plt.polar(line_pts[:, 1], line_pts[:, 0], label="No Diff IK")
    plt.legend()
    plt.subplot(222)
    plt.plot(t, line_v[:, 0], label="Radial velocity (m/s)")
    plt.plot(t, line_v[:, 1], label="Angular velocity (rad/s)")
    plt.title("Joint velocities (No Diff IK)")
    plt.legend()

    t, line_pts, line_v = plan_line_close_to_singularity(duration, timestep, 0.05)
    line_pts = fix_polar_points_for_plotting(line_pts)
    plt.subplot(223, projection="polar")
    plt.polar(line_pts[:, 1], line_pts[:, 0], label="Diff IK")
    plt.legend()
    plt.subplot(224)
    plt.plot(t, line_v[:, 0], label="Radial velocity (m/s)")
    plt.plot(t, line_v[:, 1], label="Angular velocity (rad/s)")
    plt.title("Joint velocities (Diff IK)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # plot_lines()
    plot_line_and_velocity()
