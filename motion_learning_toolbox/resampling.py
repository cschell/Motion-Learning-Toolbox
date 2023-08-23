from typing import List
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp


def resample_recording(recording: pd.DataFrame, target_fps: float, joint_names: List[str]):
    assert recording.index.inferred_type == "timedelta64", f"dataframe index has to be timedelta64, instead it was '{recording.index.inferred_type}'"

    position_columns = [f"{joint}_pos_{xyz}" for joint in joint_names for xyz in "xyz"]
    orientation_columns = [f"{joint}_rot_{xyzw}" for joint in joint_names for xyzw in "xyzw"]

    feature_columns = position_columns + orientation_columns

    features = recording[feature_columns].copy()

    assert len(features.columns) == len(feature_columns)

    mspf = 1000 / target_fps

    original_index = features.index.total_seconds() * 1000
    target_index = np.arange(original_index.min(), original_index.max(), mspf)

    interpolated_features = pd.DataFrame(index=pd.TimedeltaIndex(target_index, name="timestamp", unit="ms"))
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


