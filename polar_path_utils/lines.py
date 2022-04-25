"""Functions for planning linear paths"""
import numpy as np

from polar_path_utils.conversions import polar_to_cartesian
from polar_path_utils.diff_ik import diff_ik


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
