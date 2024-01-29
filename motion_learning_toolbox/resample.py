from typing import List
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp


def resample(data: pd.DataFrame, target_fps: float, joint_names: List[str]):
    """
    Resamples a recording DataFrame to a target frames-per-second (FPS) rate.

    :param data: A DataFrame containing the original tracking data; the DataFrame needs to have an index of type "timedelta64" (use `pd.to_timedelta` to convert integer indices).
    :param target_fps: The target frames-per-second (FPS) rate for resampling.
    :param joint_names: A list of joint names for which the data will be resampled.

    :return: A new DataFrame containing the resampled data with the specified target FPS.
    """

    assert data.index.inferred_type == "timedelta64", f"dataframe index has to be timedelta64, instead it was '{data.index.inferred_type}'"

    position_columns = [f"{joint}_pos_{xyz}" for joint in joint_names for xyz in "xyz"]
    orientation_columns = [f"{joint}_rot_{xyzw}" for joint in joint_names for xyzw in "xyzw"]

    feature_columns = position_columns + orientation_columns

    features = data[feature_columns].copy()

    assert len(features.columns) == len(feature_columns)

    mspf = 1000 / target_fps

    original_index = features.index.total_seconds() * 1000
    target_index = np.arange(original_index.min(), original_index.max(), mspf)

    interpolated_features = pd.DataFrame(index=pd.to_timedelta(target_index, unit='ms'))
    features.loc[:, position_columns] = features[position_columns].interpolate("time")

    assert not any(features[position_columns].isna().any())

    for pos_feature_name in position_columns:
        column = features[pos_feature_name]
        interpolated_features[pos_feature_name] = np.interp(x=target_index, xp=original_index, fp=column)

    for joint in joint_names:
        joint_orientation_features = [f"{joint}_rot_{c}" for c in "xyzw"]
        orientational_features = features[joint_orientation_features].dropna()
        rotations = Rotation.from_quat(orientational_features)
        dropped_na_original_index = orientational_features.index.total_seconds() * 1000
        slerp = Slerp(dropped_na_original_index, rotations)
        interpolated_features[joint_orientation_features] = slerp(target_index).as_quat()

    return interpolated_features
