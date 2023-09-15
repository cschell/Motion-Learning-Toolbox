from . import to_velocity
import pandas as pd


def to_acceleration(data: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Calculates acceleration from position and/or rotation data.

    :param data: A DataFrame or Series containing position and/or rotation data.
    :param inplace: If True, the result is stored in the original DataFrame (optional).
    :return: A DataFrame containing the calculated acceleration.
    """

    velocities = to_velocity(data, inplace=inplace)
    acceleration = to_velocity(velocities, inplace=inplace)
    return acceleration

