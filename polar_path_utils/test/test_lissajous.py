import numpy as np
import matplotlib.pyplot as plt

from polar_path_utils.lissajous import (
    plan_lissajous_polar,
)
from polar_path_utils.plotting_utils import fix_polar_points_for_plotting


def test_plan_lissajous_polar():
    # Plan a curve
    duration = 10.0
    timestep = 0.1
    max_radius = 1.0
    lissajous_t, lissajous_pts, lissajous_v = plan_lissajous_polar(
        duration,
        timestep,
        radial_freq=1,
        angular_freq=2,
        lag=0,
        max_radius=max_radius,
    )

    # Times should be correct
    assert np.isclose(lissajous_t.min(), 0.0)
    assert lissajous_t.max() < duration
    assert np.allclose(np.diff(lissajous_t).min(), timestep)
    assert np.allclose(np.diff(lissajous_t).max(), timestep)

    # Shapes should be consistent
    assert lissajous_pts.shape == lissajous_v.shape

    # Max radius and max angles should be respected
    assert np.abs(lissajous_pts[:, 0].max()) <= max_radius
    assert np.abs(lissajous_pts[:, 1].max()) <= np.pi


def plot_lissajous():
    # Plan a few curves
    duration = 10000.0
    timestep = 0.1
    max_radius = 1.0
    _, l_1_2, _ = plan_lissajous_polar(
        duration,
        timestep,
        radial_freq=1,
        angular_freq=2,
        lag=0,
        max_radius=max_radius,
    )
    l_1_2 = fix_polar_points_for_plotting(l_1_2)

    _, l_2_1, _ = plan_lissajous_polar(
        duration,
        timestep,
        radial_freq=2,
        angular_freq=1,
        lag=0,
        max_radius=max_radius,
    )
    l_2_1 = fix_polar_points_for_plotting(l_2_1)

    _, l_2_2, _ = plan_lissajous_polar(
        duration,
        timestep,
        radial_freq=2,
        angular_freq=2,
        lag=0,
        max_radius=max_radius,
    )
    l_2_2 = fix_polar_points_for_plotting(l_2_2)

    _, l_2_3, _ = plan_lissajous_polar(
        duration,
        timestep,
        radial_freq=2,
        angular_freq=3,
        lag=0,
        max_radius=max_radius,
    )
    l_2_3 = fix_polar_points_for_plotting(l_2_3)

    _, l_3_2, _ = plan_lissajous_polar(
        duration,
        timestep,
        radial_freq=3,
        angular_freq=2,
        lag=0,
        max_radius=max_radius,
    )
    l_3_2 = fix_polar_points_for_plotting(l_3_2)

    _, l_3_3, _ = plan_lissajous_polar(
        duration,
        timestep,
        radial_freq=3,
        angular_freq=3,
        lag=0,
        max_radius=max_radius,
    )
    l_3_3 = fix_polar_points_for_plotting(l_3_3)

    _, l_4_3, _ = plan_lissajous_polar(
        duration,
        timestep,
        radial_freq=4,
        angular_freq=3,
        lag=0,
        max_radius=max_radius,
    )
    l_4_3 = fix_polar_points_for_plotting(l_4_3)

    _, l_3_4, _ = plan_lissajous_polar(
        duration,
        timestep,
        radial_freq=3,
        angular_freq=4,
        lag=0,
        max_radius=max_radius,
    )
    l_3_4 = fix_polar_points_for_plotting(l_3_4)

    _, l_4_4, _ = plan_lissajous_polar(
        duration,
        timestep,
        radial_freq=4,
        angular_freq=4,
        lag=0,
        max_radius=max_radius,
    )
    l_4_4 = fix_polar_points_for_plotting(l_4_4)

    fig = plt.figure()
    ax = fig.add_subplot(331, projection="polar")
    ax.plot(l_1_2[:, 1], l_1_2[:, 0], linewidth=1, label="r_f=1, theta_f=2")
    plt.legend()
    ax = fig.add_subplot(332, projection="polar")
    ax.plot(l_2_1[:, 1], l_2_1[:, 0], linewidth=1, label="r_f=2, theta_f=1")
    plt.legend()
    ax = fig.add_subplot(333, projection="polar")
    ax.plot(l_2_2[:, 1], l_2_2[:, 0], linewidth=1, label="r_f=2, theta_f=2")
    plt.legend()
    ax = fig.add_subplot(334, projection="polar")
    ax.plot(l_2_3[:, 1], l_2_3[:, 0], linewidth=1, label="r_f=2, theta_f=3")
    plt.legend()
    ax = fig.add_subplot(335, projection="polar")
    ax.plot(l_3_2[:, 1], l_3_2[:, 0], linewidth=1, label="r_f=3, theta_f=2")
    plt.legend()
    ax = fig.add_subplot(336, projection="polar")
    ax.plot(l_3_3[:, 1], l_3_3[:, 0], linewidth=1, label="r_f=3, theta_f=3")
    plt.legend()
    ax = fig.add_subplot(337, projection="polar")
    ax.plot(l_4_3[:, 1], l_4_3[:, 0], linewidth=1, label="r_f=4, theta_f=3")
    plt.legend()
    ax = fig.add_subplot(338, projection="polar")
    ax.plot(l_3_4[:, 1], l_3_4[:, 0], linewidth=1, label="r_f=3, theta_f=4")
    plt.legend()
    ax = fig.add_subplot(339, projection="polar")
    ax.plot(l_4_4[:, 1], l_4_4[:, 0], linewidth=1, label="r_f=4, theta_f=4")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_lissajous()
