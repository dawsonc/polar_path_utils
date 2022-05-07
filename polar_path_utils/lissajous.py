"""Functions for planning lissajous figures"""
from math import gcd
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
    closed_curve: bool = False,
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
        closed_curve: if set to true, ignore duration and return a path that is long
            enough to show one full period.
    returns:
        time_waypoints: an np array of size (n_steps, 1), where each row contains the
            time of the corresponding waypoint.
        position_waypoints: an np array of size (n_steps, 2), where each row contains
            the polar coordinates of the corresponding waypoint
        velocity_waypoints: an np array of size (n_steps, 2) where each row contains
            the polar velocity of the corresponding waypoint
    """
    # Round velocities if plotting closed curves
    if closed_curve:
        radial_freq = round(radial_freq)
        angular_freq = round(angular_freq)

    # The maximum speed of the path (in polar) will be dr/dt = radial_freq * max_radius
    # and d(theta)/dt = angular_freq * pi. To respect the speed limits, we'll slow down
    # time as necessary
    max_radial_speed = radial_freq * max_radius
    max_angular_speed = angular_freq * np.pi
    slowdown = 1.0
    slowdown = np.minimum(slowdown, radial_speed_limit / max_radial_speed)
    slowdown = np.minimum(slowdown, angular_speed_limit / max_angular_speed)

    # If we need to plot only one period, then change the duration accordingly
    if closed_curve:
        duration = 2 * np.pi * gcd(int(radial_freq), int(angular_freq)) / slowdown

    # Construct the evenly-spaced time points
    t = np.arange(0.0, duration, timestep)

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
        help="The duration of the path",
    )
    parser.add_argument(
        "--closed_curve",
        action="store_true",
        help=(
            "If set, set the duration to be one period of the curve "
            "If set, both radial_frequency and angular_frequency will "
            "be rounded to the nearest integer, and --duration will be ignored."
        ),
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.1,
        help="The spacing in time between points on the path (default 0.1)",
    )
    parser.add_argument(
        "--lag",
        type=float,
        default=0.0,
        help="The lag between the radial and angular variations (default 0.0)",
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=1.0,
        help="The maximum radius of the path (default 1.0)",
    )
    parser.add_argument(
        "--radial_speed_limit",
        type=float,
        default=1.0,
        help="The maximum speed of the radial stage (default 1.0)",
    )
    parser.add_argument(
        "--angular_speed_limit",
        type=float,
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
        args.closed_curve,
    )

    if args.plot:
        position_waypoints = fix_polar_points_for_plotting(position_waypoints)
        plt.polar(position_waypoints[:, 1], position_waypoints[:, 0])
        plt.polar(
            position_waypoints[0, 1], position_waypoints[0, 0], "o", label="Start"
        )
        plt.polar(
            position_waypoints[-1, 1], position_waypoints[-1, 0], "s", label="End"
        )
        plt.legend()
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
