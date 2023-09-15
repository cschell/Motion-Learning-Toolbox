import pandas as pd


def fix_controller_mapping(data: pd.DataFrame, left_controller_name: str, right_controller_name: str, right_vector_axis: str, inplace=False) -> pd.DataFrame:
    """
    Checks if the specified target joints are swapped and swaps them if necessary. Assumes left handed coordinate system.

    :param data: A DataFrame containing tracking data; to work correctly, this data should be encoded into body relative!
    :param left_controller_name: name of the left controller.
    :param right_controller_name: name of the right controller.
    :param right_vector_axis: the axis used for comparing target joints based on their mean values.
    :param inplace: If True, the result is stored in the original DataFrame (optional).

    :return: A DataFrame where the mapping of the controllers is maintained in comparison to `right_vector_axis`.
    """
    l_mean = data[f"{left_controller_name}_pos_{right_vector_axis}"].mean()
    r_mean = data[f"{right_controller_name}_pos_{right_vector_axis}"].mean()

    if r_mean < l_mean:
        left_controller_name_rename_mapping = {c: c.replace(left_controller_name, right_controller_name) for c in data.columns if left_controller_name in c}
        right_controller_name_rename_mapping = {c: c.replace(right_controller_name, left_controller_name) for c in data.columns if right_controller_name in c}
        data = data.rename(columns={**left_controller_name_rename_mapping, **right_controller_name_rename_mapping}, inplace=inplace, copy=~inplace)
    return data
