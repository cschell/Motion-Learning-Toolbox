import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_learning_toolbox import compute_velocities_simple, compute_velocities_quats, to_acceleration, to_velocity


def test_compute_velocities_simple():
    # Load test data
    test_df = pd.read_csv("test_data.csv")[["hmd_pos_x", "hmd_pos_y", "hmd_pos_z"]]

    # Apply the function
    velocities_df = compute_velocities_simple(test_df)

    # Assumption 1: Output DataFrame has the same number of rows as the input
    assert len(velocities_df) == len(test_df)

    # Assumption 2: Output DataFrame has columns prefixed with "delta_"
    for col in test_df.columns:
        assert f"delta_{col}" in velocities_df.columns

    # Assumption 3: First row contains NaNs
    assert velocities_df.iloc[0].isna().all()

    # Assumption 4: Adding the velocity to the previous frame should return the next frame's value
    assert np.allclose(test_df.to_numpy()[:-1] + velocities_df.to_numpy()[1:], test_df.iloc[1:].to_numpy())


def test_compute_velocities_quats():
    # Load test data
    test_df = pd.read_csv("test_data.csv")[["hmd_rot_x", "hmd_rot_y", "hmd_rot_z", "hmd_rot_w"]]

    # Apply the function
    velocities_df = compute_velocities_quats(test_df)

    # Assumption 1: Output DataFrame has the same number of rows as input
    assert len(velocities_df) == len(test_df)

    # Assumption 2: Output DataFrame has columns prefixed with "delta_"
    for col in test_df.columns:
        assert f"delta_{col}" in velocities_df.columns

    # Assumption 3: Rows with original NaN quaternion values should have NaN in the output
    nan_rows = test_df.isna().any(axis=1)
    assert velocities_df.loc[nan_rows].isna().all().all()

    # Assumption 4: Rotating a frame by the following ang. velocity should yield the following frame's routation
    expected_rotations = (R.from_quat(test_df[:-1]) * R.from_quat(velocities_df[1:])).as_quat()
    actual_rotations = R.from_quat(test_df.iloc[1:]).as_quat()

    assert np.allclose(expected_rotations, actual_rotations)


def test_compute_velocities():
    # Load test data
    test_df = pd.read_csv("test_data.csv")[["hmd_rot_x", "hmd_rot_y", "hmd_rot_z", "hmd_rot_w", "hmd_pos_x", "hmd_pos_y", "hmd_pos_z"]]
    velocities_df2 = test_df.copy()

    to_velocity(velocities_df2, inplace=True)
    velocities_df = to_velocity(test_df)

    # Assumption 1: Output DataFrame has the same number of rows as input
    assert len(test_df) == len(velocities_df) == len(velocities_df2)


def test_compute_acceleration():
    # Load test data
    test_df = pd.read_csv("test_data.csv")[["hmd_rot_x", "hmd_rot_y", "hmd_rot_z", "hmd_rot_w", "hmd_pos_x", "hmd_pos_y", "hmd_pos_z"]]

    velocities_df = to_velocity(test_df)
    acceleration_df = to_acceleration(test_df)

    # Assumption 1: Output DataFrame has the same number of rows as input
    assert len(test_df) == len(velocities_df) == len(acceleration_df)

    # Assumption 2: Output DataFrames are different
    assert np.all(test_df.values == velocities_df.values, axis=1).mean() < 0.1
    assert np.all(acceleration_df.values == velocities_df.values, axis=1).mean() < 0.1
    assert np.all(acceleration_df.values == test_df.values, axis=1).mean() < 0.1
