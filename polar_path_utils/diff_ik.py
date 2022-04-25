"""Implement differential inverse kinematics"""
from typing import Any, Dict

import numpy as np
import casadi


def diff_ik(
    current_pt_polar: np.ndarray,
    desired_cartesian_velocity: np.ndarray,
    speed_limit: np.ndarray,
):
    """Solve a differential inverse kinematics problem to find the polar velocity
    that most aligns with the desired cartesian velocity, given some current polar
    configuration and speed limits.

    args:
        current_pt_polar: (2,) array of current (r, theta) configuration
        desired_cartesian_velocity: (2,) array of desired (x, y) velocities. The solver
            will attempt to find a polar velocity that moves in the same direction as
            this velocity, but may move slower.
        speed_limit: (2, 2) array where the rows include lower and upper bounds on r and
            theta, respectively
    returns:
        the polar velocity that most closely achieves the desired cartesian velocity
    """
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

    # Compute the Jacobian
    r, theta = current_pt_polar
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
    opti.subject_to(J @ v_polar == alpha * desired_cartesian_velocity)
    opti.subject_to(v_polar[0] >= speed_limit[0, 0])
    opti.subject_to(v_polar[0] <= speed_limit[0, 1])
    opti.subject_to(v_polar[1] >= speed_limit[1, 0])
    opti.subject_to(v_polar[1] <= speed_limit[1, 1])
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
    return sol.value(v_polar)
