import pandas as pd
import numpy as np
from motion_learning_toolbox import resample


def test_resample():
    # Create a test DataFrame with appropriate structure
    test_df = pd.read_csv("test_data.csv")
    test_df.index = pd.to_timedelta(test_df.timestamp, unit="ms")
    joint_names = ["hmd", "left_hand", "right_hand"]  # Replace with your joint names

    # Test various target_fps values
    target_fps_values = [30, 60, 120]

    for target_fps in target_fps_values:
        resampled_data = resample(test_df, target_fps, joint_names)

        # Assertions for resampled_data
        assert len(resampled_data) == np.ceil(test_df.index[-1].total_seconds() * target_fps)
