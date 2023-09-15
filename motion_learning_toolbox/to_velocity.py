import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def compute_velocities_simple(data: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Calculates velocities from position data using a simple differencing method.

    :param data: A DataFrame containing position data.
    :param inplace: If True, the velocities are calculated in-place.
    :return: A DataFrame containing the calculated velocities with "delta_" prefix.
    """
    velocities = data if inplace else data.copy()

    step_size = 1

    velocities.iloc[step_size:] = (
        velocities.values[step_size:] - velocities.values[:-step_size]
    )
    velocities.iloc[:step_size] = np.nan
    velocities = velocities.add_prefix("delta_")
    return velocities


def compute_velocities_quats(data: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Calculates velocities from rotation data using a simple differencing method.

    :param data: A DataFrame containing rotation data.
    :param inplace: If True, the velocities are calculated in-place.
    :return: A DataFrame containing the calculated velocities with "delta_" prefix.
    """
    velocities = data if inplace else data.copy()

    step_size = 1

    rotation_columns = [c for c in data.columns if "_rot_" in c]
    assert np.all(rotation_columns == data.columns), "rotation columns are wrong"

    joint_names = set([c[: -len("_rot_x")] for c in rotation_columns])

    for joint_name in joint_names:
        joint_rotation_names = [f"{joint_name}_rot_{c}" for c in "xyzw"]
        rotation_data = data[joint_rotation_names]

        # while computing acceleration values, we have to select the nan frames
        # (i.e., frames dismissed during the previous velocity value computation) and
        # exclude these
        nan_idxs = np.arange(len(rotation_data))[rotation_data.isna().any(axis=1)]
        rot = Rotation.from_quat(data[joint_rotation_names].fillna(0.25))
        delta_rot = rot[:-step_size].inv() * rot[step_size:]
        velocities[joint_rotation_names].iloc[step_size:] = delta_rot.as_quat()

    invalid_frames = np.concatenate(
        [
            nan_idxs,
            nan_idxs + step_size,
        ]
    )

    velocities.values[invalid_frames, :] = np.nan
    velocities = velocities.add_prefix("delta_")
    return velocities


def to_velocity(data: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Calculates velocities from position and/or rotation data.

    :param data: A DataFrame or Series containing position and/or rotation data.
    :param inplace: If True, the velocities are calculated in-place (default is False).
    :return: A DataFrame containing the calculated velocities.
    """

    velocities = data if inplace else data.copy()

    position_columns = [c for c in data.columns if "_pos_" in c]
    rotation_columns = [c for c in data.columns if "_rot_" in c]

    compute_velocities_simple(data[position_columns], inplace=True)
    compute_velocities_quats(data[rotation_columns], inplace=True)

    return velocities
