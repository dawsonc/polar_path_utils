import numpy as np
import matplotlib.pyplot as plt

from polar_path_utils.spirals import (
    plan_spiral,
)


def plan_arc(duration: float, timestep: float, wrap_angles: bool = False):
    arc_start_pt_polar = np.array([1.0, 0.0])
    arc_end_pt_polar = np.array([1.0, 2 * np.pi / 3])
    return plan_spiral(
        arc_start_pt_polar, arc_end_pt_polar, duration, timestep, wrap_angles
    )


def plan_small_arc(duration: float, timestep: float, wrap_angles: bool = False):
    small_arc_start_pt_polar = np.array([0.9, 0.0])
    small_arc_end_pt_polar = np.array([0.9, np.pi / 3])
    return plan_spiral(
        small_arc_start_pt_polar,
        small_arc_end_pt_polar,
        duration,
        timestep,
        wrap_angles,
    )


def plan_steep_spiral(duration: float, timestep: float, wrap_angles: bool = False):
    steep_spiral_start_pt_polar = np.array([0.8, 0.0])
    steep_spiral_end_pt_polar = np.array([0.4, np.pi])
    return plan_spiral(
        steep_spiral_start_pt_polar,
        steep_spiral_end_pt_polar,
        duration,
        timestep,
        wrap_angles,
    )


def plan_gradual_spiral(
    duration: float, timestep: float, rotations: int, wrap_angles: bool = False
):
    gradual_spiral_start_pt_polar = np.array([0.4, 0.0])
    gradual_spiral_end_pt_polar = np.array([0.0, rotations * 2 * np.pi])
    return plan_spiral(
        gradual_spiral_start_pt_polar,
        gradual_spiral_end_pt_polar,
        rotations * duration,
        timestep,
        wrap_angles,
    )


def test_plan_spirals():
    # Plan a spiral
    duration = 10.0
    timestep = 0.1
    rotations = 10
    spiral_t, spiral_pts, spiral_v = plan_gradual_spiral(duration, timestep, rotations)

    # Times should be correct
    assert np.isclose(spiral_t.min(), 0.0)
    assert spiral_t.max() < rotations * duration
    assert np.allclose(np.diff(spiral_t).min(), timestep)
    assert np.allclose(np.diff(spiral_t).max(), timestep)

    # Velocity should be constant
    acceleration = np.diff(spiral_v, axis=0)
    assert np.allclose(acceleration, np.zeros_like(acceleration))

    # Test whether angles wrap correctly
    _, spiral_pts, _ = plan_gradual_spiral(
        duration, timestep, rotations, wrap_angles=True
    )
    assert spiral_pts.max() <= 2 * np.pi
    assert spiral_pts.min() >= 0.0


def plot_spirals():
    # use the same duration and timestep for all (except the gradual spiral, which
    # uses this duration for each of 10 revolutions)
    duration = 10.0
    timestep = 0.1

    # Plan an arc
    _, arc_pts, _ = plan_arc(duration, timestep)

    # Plan a smaller arc
    _, small_arc_pts, _ = plan_small_arc(duration, timestep)

    # Plan a steep spiral
    _, steep_spiral_pts, _ = plan_steep_spiral(duration, timestep)

    # Plan a gradual spiral
    rotations = 10
    _, gradual_spiral_pts, _ = plan_gradual_spiral(duration, timestep, rotations)

    plt.polar(arc_pts[:, 1], arc_pts[:, 0], linewidth=3, label="Large arc")
    plt.polar(small_arc_pts[:, 1], small_arc_pts[:, 0], linewidth=3, label="Small arc")
    plt.polar(
        steep_spiral_pts[:, 1],
        steep_spiral_pts[:, 0],
        linewidth=3,
        label="Steep spiral",
    )
    plt.polar(
        gradual_spiral_pts[:, 1],
        gradual_spiral_pts[:, 0],
        linewidth=3,
        label="Gradual spiral",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_spirals()
