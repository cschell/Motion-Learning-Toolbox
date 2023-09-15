import pandas as pd
import numpy as np
from motion_learning_toolbox import canonicalize_quaternions


def test_canonicalize_quaternions():
    # Load test data
    test_df = pd.read_csv("test_data.csv")

    # The joint names for which quaternion data exists in test_df
    joint_names = ["hmd", "left_hand", "right_hand"]

    # Apply the function
    canonical_df = canonicalize_quaternions(test_df, joint_names)

    # Assumption 1: w-component should be non-negative
    for joint in joint_names:
        w_column = f"{joint}_rot_w"
        assert all(canonical_df[w_column] >= 0)

    # Assumption 2: Quaternions should be normalized
    for joint in joint_names:
        quat_cols = [f"{joint}_rot_{xyzw}" for xyzw in "wxyz"]
        norms = np.linalg.norm(canonical_df[quat_cols].values, axis=1)
        assert all(np.isclose(norms, 1, atol=1e-6))

    # Assumption 3: Function should not alter unrelated columns
    unrelated_columns = [col for col in test_df.columns if all(joint not in col for joint in joint_names)]
    assert test_df[unrelated_columns].equals(canonical_df[unrelated_columns])
