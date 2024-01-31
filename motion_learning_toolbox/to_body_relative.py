from typing import Dict, List
import quaternionic
import numpy as np
import pandas as pd


def quaternion_composition(quaternion_array1, quaternion_array2):
    w1, x1, y1, z1 = (
        quaternion_array1[:, 0],
        quaternion_array1[:, 1],
        quaternion_array1[:, 2],
        quaternion_array1[:, 3],
    )
    w2, x2, y2, z2 = (
        quaternion_array2[:, 0],
        quaternion_array2[:, 1],
        quaternion_array2[:, 2],
        quaternion_array2[:, 3],
    )

    w_composed = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_composed = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_composed = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_composed = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    composed_quaternions = quaternionic.array(np.column_stack((w_composed, x_composed, y_composed, z_composed))).normalized
    return composed_quaternions


def to_body_relative(
    frames: pd.DataFrame,
    target_joints: List[str],
    coordinate_system: Dict[str, str],
    reference_joint="head",
):
    """
    Transforms position and rotation data into a body-relative coordinate system.

    :param frames: A DataFrame or Series containing position and/or rotation data.
    :param target_joints: A list of joints to be transformed.
    :param coordinate_system: A dictionary specifying the coordinate system for the transformation.
    :param reference_joint: The reference joint used as the origin of the body-relative coordinate system (default is "head").
    """
    reference_pos_columns = [f"{reference_joint}_pos_{xyz}" for xyz in "xyz"]
    target_dtype = frames[reference_pos_columns[0]].to_numpy().dtype
    target_dtype = target_dtype if np.issubdtype(target_dtype, np.floating) else "float32"
    min_float32_dtype = "float32" if target_dtype == np.float16 else target_dtype

    FORWARD = "xyz".index(coordinate_system["forward"])
    RIGHT = "xyz".index(coordinate_system["right"])
    UP = "xyz".index(coordinate_system["up"])

    assert FORWARD != RIGHT != UP

    FORWARD_DIRECTION = np.identity(3, dtype=target_dtype)[FORWARD]
    UP_DIRECTION = np.identity(3, dtype=target_dtype)[UP]

    num_samples = len(frames)

    ## parse rotations of the reference joint (the head)
    reference_rotation_names = [f"{reference_joint}_rot_{c}" for c in "wxyz"]
    reference_rotations = quaternionic.array(frames[reference_rotation_names], dtype=target_dtype).normalized.astype(min_float32_dtype)

    ## retrieve projection of viewing direction of the reference joint on
    ## the horizontal plane by first applying the head rotation onto the
    ## forward vector and then zeroing out the UP axis
    horizontal_plane_projections = reference_rotations.rotate(FORWARD_DIRECTION)

    horizontal_plane_projections[:, UP] = 0

    rotations_around_up_axis = np.arccos((horizontal_plane_projections @ FORWARD_DIRECTION) / (np.linalg.norm(FORWARD_DIRECTION) * np.linalg.norm(horizontal_plane_projections, axis=1)))

    ## compute correction rotation
    # find out into which direction the vectors have to be rotated
    correction_rotation_directions = -np.sign(horizontal_plane_projections[:, RIGHT])

    # build euler angle rotation vector for rotation around UP axis
    # (usage of `.from_axis_angle` feels a bit hacky, but that's easier than building
    # a rotation matrix from scratch)
    correction_rotations_raw = np.zeros((num_samples, 3), dtype=target_dtype)
    correction_rotations_raw[:, UP] = correction_rotation_directions * rotations_around_up_axis
    correction_rotations = quaternionic.array.from_axis_angle(correction_rotations_raw).astype(min_float32_dtype)

    ## apply correction positions and rotations
    relative_positions_and_rotations = pd.DataFrame()

    for joint_name in target_joints:  # => joint_name is either `right_hand` or `left_hand`
        # apply rotations to position vector of `joint_name`
        joint_position_names = [f"{joint_name}_pos_{c}" for c in "xyz"]

        shifted_positions = frames[joint_position_names].values - frames[reference_pos_columns].values

        shifted_and_rotated_positions = np.einsum("ijk,ik->ij", correction_rotations.to_rotation_matrix, shifted_positions)
        relative_positions_and_rotations[joint_position_names] = shifted_and_rotated_positions

        # rotate the world rotation of `joint_name` by the correction rotation and save quaternion representations
        joint_rotation_names = [f"{joint_name}_rot_{c}" for c in "wxyz"]

        sr_rotations = quaternionic.array(frames[joint_rotation_names]).astype(target_dtype)
        br_rotations = quaternion_composition(correction_rotations.ndarray, sr_rotations.ndarray)

        relative_positions_and_rotations[joint_rotation_names] = br_rotations.normalized.astype(target_dtype).ndarray

    # add horizontal rotations of reference joint
    relative_positions_and_rotations[reference_rotation_names] = (correction_rotations * reference_rotations).normalized.astype(target_dtype).ndarray

    return relative_positions_and_rotations
