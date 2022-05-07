"""Functions for planning lissajous figures"""
import sys

import argparse
import numpy as np
import matplotlib.pyplot as plt

from polar_path_utils.file_utils import save_path_to_csv
from polar_path_utils.plotting_utils import fix_polar_points_for_plotting


def plan_lissajous_polar(
    duration: float,
    timestep: float,
    radial_freq: float = 1,
    angular_freq: float = 2,
    lag: float = np.pi / 2.0,
    max_radius: float = 1.0,
    radial_speed_limit: float = 0.1,
    angular_speed_limit: float = np.pi / 10,
):
    """
    Plan a path that makes a Lissajous curve (en.wikipedia.org/wiki/Lissajous_curve),
    where r and theta are functions of time:

        r = max_radius * sin(radial_freq * t)
        theta = pi * sin(angular_freq * t + lag)

    The figure will start at the origin: r = 0, theta = pi * sin(lag)

    The returned path is a sequence of n_steps waypoints (points in polar coordinates)
    spaced evenly in time along the path.

    args:
        start_pt_polar: an np array of two elements [r, theta], representing polar
            coordinates of the start point of the spiral.
        duration: the amount of time the trajectory will take to reach the end point
        timestep: the amount of time between successive waypoints.
        radial_freq: the frequency of the radial variation.
        angular_freq: the frequency of the angular variation.
        lag: the delay between the angular and radial variation.
        max_radius: the maximum radius of the curve.
        radial_speed_limit: speed limit for radial motion (m/s)
        angular_speed_limit: speed limit for angular motion (radians/s)
    returns:
        time_waypoints: an np array of size (n_steps, 1), where each row contains the
            time of the corresponding waypoint.
        position_waypoints: an np array of size (n_steps, 2), where each row contains
            the polar coordinates of the corresponding waypoint
        velocity_waypoints: an np array of size (n_steps, 2) where each row contains
            the polar velocity of the corresponding waypoint
    """
    # Construct the evenly-spaced time points
    t = np.arange(0.0, duration, timestep)

    # The maximum speed of the path (in polar) will be dr/dt = radial_freq * max_radius
    # and d(theta)/dt = angular_freq * pi. To respect the speed limits, we'll slow down
    # time as necessary
    max_radial_speed = radial_freq * max_radius
    max_angular_speed = angular_freq * np.pi
    slowdown = 1.0
    slowdown = np.minimum(slowdown, radial_speed_limit / max_radial_speed)
    slowdown = np.minimum(slowdown, angular_speed_limit / max_angular_speed)

    # Make the waypoints from the generating functions
    position_waypoints = np.zeros((t.shape[0], 2))
    position_waypoints[:, 0] = max_radius * np.sin(radial_freq * t * slowdown)
    position_waypoints[:, 1] = np.pi * np.sin(
        angular_freq * t * slowdown + lag * slowdown
    )

    # Make velocity waypoints by analytically differentiating
    velocity_waypoints = np.zeros_like(position_waypoints)
    velocity_waypoints[:, 0] = radial_freq * max_radius * np.cos(radial_freq * t)
    velocity_waypoints[:, 1] = angular_freq * np.pi * np.cos(angular_freq * t + lag)

    return t, position_waypoints, velocity_waypoints


def lissajous_cli():
    """Define a command-line interface for plotting lines"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Path to the file where you want to save the path. "
            "If not provided, defaults to printing the path to stdout"
        ),
    )
    parser.add_argument(
        "--radial_frequency",
        type=float,
        required=True,
        help="The radial frequency of the figure",
    )
    parser.add_argument(
        "--angular_frequency",
        type=float,
        required=True,
        help="The angular frequency of the path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        help="The duration of the path",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        nargs=1,
        default=0.1,
        help="The spacing in time between points on the path (default 0.1)",
    )
    parser.add_argument(
        "--lag",
        type=float,
        nargs=1,
        default=0.0,
        help="The lag between the radial and angular variations (default 0.0)",
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        nargs=1,
        default=1.0,
        help="The maximum radius of the path (default 1.0)",
    )
    parser.add_argument(
        "--radial_speed_limit",
        type=float,
        nargs=1,
        default=1.0,
        help="The maximum speed of the radial stage (default 1.0)",
    )
    parser.add_argument(
        "--angular_speed_limit",
        type=float,
        nargs=1,
        default=np.pi,
        help="The maximum speed of the angular stage (default pi)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, plot the path",
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="If set, convert all angles to degrees",
    )
    args = parser.parse_args()

    # Plan the path
    time_waypoints, position_waypoints, velocity_waypoints = plan_lissajous_polar(
        args.duration,
        args.timestep,
        args.radial_frequency,
        args.angular_frequency,
        args.lag,
        args.max_radius,
        args.radial_speed_limit,
        args.angular_speed_limit,
    )

    if args.plot:
        position_waypoints = fix_polar_points_for_plotting(position_waypoints)
        plt.polar(position_waypoints[:, 1], position_waypoints[:, 0])
        plt.show()

    # Save
    file = sys.stdout.buffer
    if args.save_path is not None:
        file = args.save_path

    save_path_to_csv(
        file, time_waypoints, position_waypoints, velocity_waypoints, args.degrees
    )


if __name__ == "__main__":
    lissajous_cli()
