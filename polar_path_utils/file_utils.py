"""Utilities for saving paths to files"""
import numpy as np


def save_path_to_csv(
    filename: str,
    time_waypoints: np.ndarray,
    position_waypoints: np.ndarray,
    velocity_waypoints: np.ndarray,
    degrees: bool = False
):
    """Save the given path to a CSV with the given name.

    args:
        filename: the location to save the path
        time_waypoints, position_waypoints, velocity_waypoints: the path to save
        degrees: if True, convert angles and angular velocities from radians to degrees.
    """
    # Convert the angular data if necessary
    if degrees:
        position_waypoints = position_waypoints.copy()
        velocity_waypoints = velocity_waypoints.copy()
        position_waypoints[:, 1] *= 180.0 / np.pi
        velocity_waypoints[:, 1] *= 180.0 / np.pi

    # Concatenate the data
    data = np.concatenate(
        (time_waypoints.reshape(-1, 1), position_waypoints, velocity_waypoints), axis=-1
    )
    assert data.shape[0] == time_waypoints.shape[0]

    # Save to a CSV
    header = "t,r,theta,v_r,v_theta"
    np.savetxt(filename, data, header=header, delimiter=",", fmt="%.5f")
