"""Functions for planning spiral paths"""
import numpy as np


def plan_spiral(
    start_pt_polar: np.ndarray,
    end_pt_polar: np.ndarray,
    duration: float,
    timestep: float,
    wrap_angles: bool = False,
):
    """
    Plan a path that linearly interpolates between two polar points; this will look like
    a spiral if the points have different radii, and the arc of a circle if the points
    have the same radii.

    The path is returned as a sequence of n_steps waypoints (points in polar coordinates)
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
