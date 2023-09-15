from typing import List
import numpy as np
import pandas as pd
import quaternionic


def canonicalize_quaternions(data: pd.DataFrame, joint_names: List[str], inplace=False) -> pd.DataFrame:
    """
    Canonicalize the quaternions in the DataFrame for a given list of joint names.
    
    The function modifies the quaternion rotation data to follow a canonical form. 
    Specifically, it ensures that the w-component of the quaternion is non-negative and that the returned quaternion is normalized.
    This provides a unique representation for each rotation, which is useful for machine learning applications.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing quaternion rotation data. The DataFrame should have columns in the format "{joint_name}_rot_{wxyz}" for each joint.
        
    joint_names : List[str]
        List of joint names for which quaternion data should be canonicalized. Each joint name should correspond to quaternion columns in the DataFrame.
        
    inplace : bool, optional
        If True, modifies the DataFrame in place. Otherwise, returns a new DataFrame with the canonicalized quaternions. Default is False.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with canonicalized quaternion data for the joints specified by `joint_names`.
    """
    if not inplace:
        data = data.copy()

    for joint in joint_names:
        rotation_column_names = [f"{joint}_rot_{xyzw}" for xyzw in "wxyz"]
        canonicalized_quaternion = data[rotation_column_names].to_numpy() * np.sign(data[rotation_column_names[0]]).to_numpy()[:, None]
        data[rotation_column_names] = quaternionic.array(canonicalized_quaternion).normalized.ndarray

    return data
