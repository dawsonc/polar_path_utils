"""Functions for planning lissajous figures"""
import numpy as np


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
