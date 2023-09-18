import pandas as pd
import numpy as np
from motion_learning_toolbox import fix_controller_mapping
import pytest


@pytest.mark.parametrize("switched", [False, True])
def test_fix_controller_mapping(switched: bool):
    np.random.seed(42)
    n_samples = 1000
    hmd_positions_dict = {
        "hmd_pos_x": np.random.normal(loc=0, scale=0.5, size=n_samples),
        "hmd_pos_y": np.random.normal(loc=20, scale=3, size=n_samples),
        "hmd_pos_z": np.random.normal(loc=3, scale=1, size=n_samples),
        "left_hand_pos_x": np.random.normal(
            loc=-1 * -((switched - 0.5) * 2), scale=3, size=n_samples
        ),
        "left_hand_pos_y": np.random.normal(loc=20, scale=3, size=n_samples),
        "left_hand_pos_z": np.random.normal(loc=3, scale=1, size=n_samples),
        "right_hand_pos_x": np.random.normal(
            loc=1 * -((switched - 0.5) * 2), scale=3, size=n_samples
        ),
        "right_hand_pos_y": np.random.normal(loc=20, scale=3, size=n_samples),
        "right_hand_pos_z": np.random.normal(loc=3, scale=1, size=n_samples),
    }

    hmd_positions = pd.DataFrame(hmd_positions_dict)

    # Define the coordinate system for transformation
    coordinate_system = {"forward": "z", "right": "x", "up": "y"}

    fixed_positions = fix_controller_mapping(
        hmd_positions,
        left_controller_name="left_hand",
        right_controller_name="right_hand",
        coordinate_system=coordinate_system,
        is_br=True,
        inplace=False,
    )

    if switched:
        rename_mapping = {
            "left_hand_pos_x": "right_hand_pos_x",
            "left_hand_pos_y": "right_hand_pos_y",
            "left_hand_pos_z": "right_hand_pos_z",
            "right_hand_pos_x": "left_hand_pos_x",
            "right_hand_pos_y": "left_hand_pos_y",
            "right_hand_pos_z": "left_hand_pos_z",
        }
    else:
        rename_mapping = {}

    assert np.all(
        hmd_positions
        == fixed_positions.rename(columns=rename_mapping)[hmd_positions.columns]
    )
