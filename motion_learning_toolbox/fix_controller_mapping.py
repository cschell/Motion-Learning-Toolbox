from typing import Dict
import pandas as pd

from motion_learning_toolbox.to_body_relative import to_body_relative


def fix_controller_mapping(
    data: pd.DataFrame,
    left_controller_name: str,
    right_controller_name: str,
    coordinate_system: Dict[str, str],
    reference_joint=None,
    is_br=False,
    inplace=False,
) -> pd.DataFrame:
    """
    Checks if the specified target joints are swapped and swaps them if necessary. Assumes left handed coordinate system.

    :param data: A DataFrame containing tracking data; to work correctly, this data should be encoded into body relative!
    :param left_controller_name: name of the left controller.
    :param right_controller_name: name of the right controller.
    :param right_vector_axis: the axis used for comparing target joints based on their mean values.
    :param coordinate_system: A dictionary specifying the used coordinate system.
    :param reference_joint: The reference joint used as the origin of the body-relative coordinate system.
    :param is_br: the implementation assumes that `data` has not yet been converted to body-relative, which is a necessary requirement. Should the data already be in BR, set is_br to True.
    :param inplace: If True, the result is stored in the original DataFrame (optional).

    :return: A DataFrame where the mapping of the controllers is maintained in comparison to `right_vector_axis`.
    """
    if not is_br:
        br_data = to_body_relative(
            data,
            target_joints=[left_controller_name, right_controller_name],
            coordinate_system=coordinate_system,
            reference_joint=reference_joint,
        )
    else:
        br_data = data

    l_mean = br_data[f"{left_controller_name}_pos_{coordinate_system['right']}"].mean()
    r_mean = br_data[f"{right_controller_name}_pos_{coordinate_system['right']}"].mean()

    if r_mean < l_mean:
        left_controller_name_rename_mapping = {
            c: c.replace(left_controller_name, right_controller_name)
            for c in data.columns
            if left_controller_name in c
        }
        right_controller_name_rename_mapping = {
            c: c.replace(right_controller_name, left_controller_name)
            for c in data.columns
            if right_controller_name in c
        }
        data = data.rename(
            columns={
                **left_controller_name_rename_mapping,
                **right_controller_name_rename_mapping,
            },
            inplace=inplace,
            copy=~inplace,
        )
    return data
