"""Functions for planning linear paths"""
from typing import Any, Dict

import numpy as np
import casadi


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
    r, theta = end_pt_polar
    end_pt_cartesian = np.array([r * np.cos(theta), r * np.sin(theta)])

    # Integrate along the trajectory to solve for the waypoints in polar space. This
    # is kind of like a differential IK controller.
    position_waypoints[0, :] = start_pt_polar
    for i in range(1, time_waypoints.shape[0]):
        # Get the most recent waypoint along the trajectory
        last_waypoint_polar = position_waypoints[i - 1]
        r = last_waypoint_polar[0]
        theta = last_waypoint_polar[1]
        t = time_waypoints[i]

        # The position of the tool is (r * cos(theta), r * sin(theta)), so the Jacobian
        # is J =
        #
        #   [cos(theta), -r * sin(theta)]
        #   [sin(theta), r * cos(theta)]
        #
        # Given the Jacobian and some polar velocity, the cartesian velocity is
        #
        #   v_cartesian = J @ v_polar  (matrix multiplication)
        #
        # So we can solve for the polar velocity that gets us closest to the desired
        # cartesian velocity using a constrained optimization. Closest
        # is defined by smallest deviation from the desired direction.
        # The optimization problem is formally defined as:
        #
        #   solve for alpha (scalar) and v_polar (2D vector)
        #   to maximize alpha
        #   such that
        #             J @ v_polar = alpha * V_cartesian
        #             v_polar obeys speed limits
        #             0 <= alpha <= 1
        #
        # This will mean that the path will slow down if it has to pass near a
        # kinematic singularity (where the Jacobian becomes near-singular)
        #
        # See https://manipulation.csail.mit.edu/pick.html#diff_ik_w_constraints for
        # more info.
        start_pt_cartesian = np.array([r * np.cos(theta), r * np.sin(theta)])
        v_cartesian = (end_pt_cartesian - start_pt_cartesian) / max(duration - t, 1e-2)

        # Compute the Jacobian
        J = np.array(
            [
                [np.cos(theta), -r * np.sin(theta)],
                [np.sin(theta), r * np.cos(theta)],
            ]
        )

        # Define the optimization problem
        opti = casadi.Opti()

        v_polar = opti.variable(2)
        alpha = opti.variable()

        opti.minimize(-alpha)  # minimize negative = maximize
        opti.subject_to(J @ v_polar == alpha * v_cartesian)
        opti.subject_to(v_polar[0] >= -radial_speed_limit)
        opti.subject_to(v_polar[0] <= radial_speed_limit)
        opti.subject_to(v_polar[1] >= -angular_speed_limit)
        opti.subject_to(v_polar[1] <= angular_speed_limit)
        opti.subject_to(0.0 <= alpha)
        opti.subject_to(alpha <= 1.0)

        # Set solver options and solve
        p_opts: Dict[str, Any] = {"expand": True}
        s_opts: Dict[str, Any] = {"max_iter": 1000}
        quiet = True
        if quiet:
            p_opts["print_time"] = 0
            s_opts["print_level"] = 0
            s_opts["sb"] = "yes"
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

        # Extract the velocity from the result
        velocity_waypoints[i] = sol.value(v_polar)

        # Get the next waypoint by integrating this velocity
        position_waypoints[i] = last_waypoint_polar + timestep * velocity_waypoints[i]

    # Return the constructed path
    return time_waypoints, position_waypoints, velocity_waypoints


def danger_line(
    start_pt_polar: np.ndarray,
    end_pt_polar: np.ndarray,
    duration: float,
    timestep: float,
    radial_speed_limit: float = 1.0,
    angular_speed_limit: float = np.pi,
):
    """Convert a line to polar coordinates in the dangerous way.

    args:
        start_pt_polar: an np array of two elements [r, theta], representing polar
            coordinates of the start point of the line.
        end_pt_polar: an np array of two elements [r, theta], representing polar
            coordinates of the end point of the line.
        duration: the amount of time the trajectory will take to reach the end point
        timestep: the amount of time between successive waypoints.
    returns:
        time_waypoints: an np array of size (n_steps, 1), where each row contains the
            time of the corresponding waypoint.
        position_waypoints: an np array of size (n_steps, 2), where each row contains
            the polar coordinates of the corresponding waypoint
        velocity_waypoints: an np array of size (n_steps, 2) where each row contains
            the polar velocity of the corresponding waypoint
    """
    time_waypoints = np.arange(0.0, duration, timestep)

    # Linearly interpolate in cartesian space
    r, theta = start_pt_polar
    start_pt_cartesian = np.array([r * np.cos(theta), r * np.sin(theta)])
    r, theta = end_pt_polar
    end_pt_cartesian = np.array([r * np.cos(theta), r * np.sin(theta)])

    # Linearly interpolate x coordinates
    x = np.interp(
        time_waypoints,
        [0.0, duration],
        [start_pt_cartesian[0], end_pt_cartesian[0]],
    )
    y = np.interp(
        time_waypoints,
        [0.0, duration],
        [start_pt_cartesian[1], end_pt_cartesian[1]],
    )

    # Convert these back to polar
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    position_waypoints = np.hstack((radius.reshape(-1, 1), theta.reshape(-1, 1)))

    # Compute the velocity by differentiating
    velocity_waypoints = np.diff(position_waypoints, axis=0) / timestep
    # pad the velocity
    velocity_waypoints = np.concatenate(
        (velocity_waypoints, velocity_waypoints[-1, :].reshape(1, -1))
    )

    return time_waypoints, position_waypoints, velocity_waypoints
