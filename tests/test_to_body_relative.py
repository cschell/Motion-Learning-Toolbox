import pandas as pd
import numpy as np

from motion_learning_toolbox import to_body_relative

from scipy.spatial.transform import Rotation


def test_to_body_relative():
    # Load the test data
    test_df = pd.read_csv("test_data.csv")
    test_df.index = pd.to_timedelta(test_df.timestamp, unit="ms")
    target_joints = ["left_hand", "right_hand"]

    # Define the coordinate system for transformation
    coordinate_system = {"forward": "z", "right": "x", "up": "y"}

    # Specify the reference joint
    reference_joint = "hmd"

    # Perform the transformation
    transformed_data = to_body_relative(
        test_df, target_joints, coordinate_system, reference_joint
    )

    # Assertions for transformed_data
    assert len(transformed_data) == len(
        test_df
    )  # Ensure the number of samples remains the same
    assert transformed_data.isna().any().sum() == 0


def test_br_rotation_of_rotations_and_positions():
    left_hand_position = [123, 23, 1]
    hmd_position = [123, 23, 0]
    right_hand_position = [123, 23, -1]

    left_hand_orientation = Rotation.from_euler(
        "xyz", [0, 90, 30], degrees=True
    ).as_quat()
    right_hand_orientation = Rotation.from_euler(
        "xyz", [0, 90, 40], degrees=True
    ).as_quat()
    hmd_orientation = Rotation.from_euler("xyz", [0, 90, 50], degrees=True).as_quat()

    test_data = [
        [
            *left_hand_position,
            *right_hand_position,
            *hmd_position,
            *left_hand_orientation,
            *right_hand_orientation,
            *hmd_orientation,
        ]
    ]
    joints = ["left_hand", "right_hand", "hmd"]

    test_df = pd.DataFrame(
        test_data,
        columns=[f"{joint}_pos_{xyz}" for joint in joints for xyz in "xyz"]
        + [f"{joint}_rot_{xyzw}" for joint in joints for xyzw in "xyzw"],
    )

    transformed_data = to_body_relative(
        test_df,
        reference_joint="hmd",
        target_joints=["left_hand", "right_hand"],
        coordinate_system={"forward": "z", "right": "x", "up": "y"},
    )

    expected_left_hand_orientation = Rotation.from_euler(
        "xyz", [-30, 0, 0], degrees=True
    ).as_quat()
    expected_right_hand_orientation = Rotation.from_euler(
        "xyz", [-40, 0, 0], degrees=True
    ).as_quat()
    assert np.allclose(
        transformed_data[
            [
                "left_hand_rot_x",
                "left_hand_rot_y",
                "left_hand_rot_z",
                "left_hand_rot_w",
                "right_hand_rot_x",
                "right_hand_rot_y",
                "right_hand_rot_z",
                "right_hand_rot_w",
            ]
        ],
        [[*expected_left_hand_orientation, *expected_right_hand_orientation]],
    )

    expected_left_hand_position = [-1, 0, 0]
    expected_right_hand_position = [1, 0, 0]
    assert np.allclose(
        transformed_data[
            [
                "left_hand_pos_x",
                "left_hand_pos_y",
                "left_hand_pos_z",
                "right_hand_pos_x",
                "right_hand_pos_y",
                "right_hand_pos_z",
            ]
        ],
        [[*expected_left_hand_position, *expected_right_hand_position]],
    )
