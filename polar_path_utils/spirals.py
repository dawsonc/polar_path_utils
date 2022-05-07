"""Functions for planning spiral paths"""
import sys

import argparse
import numpy as np
import matplotlib.pyplot as plt

from polar_path_utils.file_utils import save_path_to_csv
from polar_path_utils.plotting_utils import fix_polar_points_for_plotting


def plan_spiral(
    start_pt_polar: np.ndarray,
    end_pt_polar: np.ndarray,
    duration: float,
    timestep: float,
    wrap_angles: bool = False,
):
    """
    Plan a path that interpolates between two polar points linearly in joint space;
    this will look like a spiral if the points have different radii, and the arc of a
    circle if the points have the same radii.

    The returned path is a sequence of n_steps waypoints (points in polar coordinates)
    spaced evenly in time along the path.

    By default, returns a path where the polar angle can range from -infinity to
    +infinity (increasing just wraps around the circle), but setting wrap_angles to True
    will confine all angles to be between 0 and 2*pi.

    args:
        start_pt_polar: an np array of two elements [r, theta], representing polar
            coordinates of the start point of the spiral.
        end_pt_polar: an np array of two elements [r, theta], representing polar
            coordinates of the end point of the spiral.
        duration: the amount of time the trajectory will take to reach the end point
        timestep: the amount of time between successive waypoints.
        wrap_angles: if True, wrap angles to be within 0 and 2*pi.
    returns:
        time_waypoints: an np array of size (n_steps, 1), where each row contains the
            time of the corresponding waypoint.
        position_waypoints: an np array of size (n_steps, 2), where each row contains
            the polar coordinates of the corresponding waypoint
        velocity_waypoints: an np array of size (n_steps, 2) where each row contains
            the polar velocity of the corresponding waypoint
    """
    # Construct the evenly-spaced time points
    time_waypoints = np.arange(0.0, duration, timestep)

    # Create the spiral by linearly interpolating between the two points
    # First do this for the radius
    radius_waypoints = np.interp(
        time_waypoints,
        [0.0, duration],
        [start_pt_polar[0], end_pt_polar[0]],
    )
    # Then do the angle
    angle_waypoints = np.interp(
        time_waypoints,
        [0.0, duration],
        [start_pt_polar[1], end_pt_polar[1]],
    )
    # And assemble into the waypoints
    position_waypoints = np.hstack(
        (radius_waypoints.reshape(-1, 1), angle_waypoints.reshape(-1, 1))
    )

    # The velocity will be constant along the spiral
    radius_velocity = (end_pt_polar[0] - start_pt_polar[0]) / duration
    angle_velocity = (end_pt_polar[1] - start_pt_polar[1]) / duration
    velocity_waypoints = np.zeros_like(position_waypoints)
    velocity_waypoints[:, 0] = radius_velocity
    velocity_waypoints[:, 1] = angle_velocity

    # Wrap the angles if necessary
    if wrap_angles:
        angles = position_waypoints[:, 1]
        # This wraps all angles to [-pi, pi]
        position_waypoints[:, 1] = np.arctan2(np.sin(angles), np.cos(angles))
        # Shift to [0, 2*pi]
        position_waypoints[:, 1] += np.pi

    return time_waypoints, position_waypoints, velocity_waypoints


def spirals_cli():
    """Define a command-line interface for plotting spirals"""
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
        "--start_pt_polar",
        type=float,
        nargs=2,
        required=True,
        help="The radius and angle of the starting point",
    )
    parser.add_argument(
        "--end_pt_polar",
        type=float,
        nargs=2,
        required=True,
        help="The radius and angle of the ending point",
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
        default=0.1,
        help="The spacing in time between points on the path (default 0.1)",
    )
    parser.add_argument(
        "--wrap_angles",
        action="store_true",
        help="If set, wrap all angles to be between 0 and 2*pi",
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
    time_waypoints, position_waypoints, velocity_waypoints = plan_spiral(
        np.array(args.start_pt_polar),
        np.array(args.end_pt_polar),
        args.duration,
        args.timestep,
        args.wrap_angles,
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
    spirals_cli()
