"""Functions for planning linear paths"""
import sys

import argparse
import numpy as np
import matplotlib.pyplot as plt

from polar_path_utils.conversions import polar_to_cartesian
from polar_path_utils.diff_ik import diff_ik
from polar_path_utils.file_utils import save_path_to_csv
from polar_path_utils.plotting_utils import fix_polar_points_for_plotting


def plan_line(
    start_pt_polar: np.ndarray,
    end_pt_polar: np.ndarray,
    duration: float,
    timestep: float,
    radial_speed_limit: float = 1.0,
    angular_speed_limit: float = np.pi,
):
    """
    Plan a path that interpolates between two points linearly in tool space;
    this will look like a line.

    The returned path is a sequence of n_steps waypoints (points in polar coordinates)
    spaced evenly in time along the path.

    args:
        start_pt_polar: an np array of two elements [r, theta], representing polar
            coordinates of the start point of the line.
        end_pt_polar: an np array of two elements [r, theta], representing polar
            coordinates of the end point of the line.
        duration: the amount of time the trajectory will take to reach the end point
        timestep: the amount of time between successive waypoints.
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
    time_waypoints = np.arange(0.0, duration, timestep)

    # And create some arrays to hold the position and velocity waypoints
    position_waypoints = np.zeros((time_waypoints.shape[0], 2))
    velocity_waypoints = np.zeros((time_waypoints.shape[0], 2))

    # To construct a path that moves at constant velocity in tool space, we need to
    # figure out how to map linear to polar velocity at each point along the path.
    # We'll do this step-by-step, integrating the trajectory as we go

    # We'll need to reference the goal point as we go
    end_pt_cartesian = polar_to_cartesian(end_pt_polar)

    # Integrate along the trajectory to solve for the waypoints in polar space. This
    # is kind of like a differential IK controller.
    position_waypoints[0, :] = start_pt_polar
    for i in range(1, time_waypoints.shape[0]):
        # Get the most recent waypoint along the trajectory
        last_waypoint_polar = position_waypoints[i - 1]
        t = time_waypoints[i]

        start_pt_cartesian = polar_to_cartesian(last_waypoint_polar)
        v_cartesian = (end_pt_cartesian - start_pt_cartesian) / max(duration - t, 1e-2)

        # Solve a differential IK problem to find a polar velocity that gets close to
        # this desired cartesian velocity
        speed_limit = np.array(
            [
                [-radial_speed_limit, radial_speed_limit],
                [-angular_speed_limit, angular_speed_limit],
            ]
        )
        velocity_waypoints[i] = diff_ik(last_waypoint_polar, v_cartesian, speed_limit)

        # Get the next waypoint by integrating this velocity
        position_waypoints[i] = last_waypoint_polar + timestep * velocity_waypoints[i]

    # Return the constructed path
    return time_waypoints, position_waypoints, velocity_waypoints


def lines_cli():
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
    time_waypoints, position_waypoints, velocity_waypoints = plan_line(
        np.array(args.start_pt_polar),
        np.array(args.end_pt_polar),
        args.duration,
        args.timestep,
        args.radial_speed_limit,
        args.angular_speed_limit,
    )

    if args.plot:
        plotting_positions = fix_polar_points_for_plotting(position_waypoints)
        plt.polar(plotting_positions[:, 1], plotting_positions[:, 0])
        plt.polar(
            plotting_positions[0, 1], plotting_positions[0, 0], "o", label="Start"
        )
        plt.polar(
            plotting_positions[-1, 1], plotting_positions[-1, 0], "s", label="End"
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
    lines_cli()
