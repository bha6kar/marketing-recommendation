from typing import List

from keras import backend as K
from pandas import DataFrame


def get_unique_values(df: DataFrame, column: str) -> List:
    """
    Print unique values of a specified column in a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - column (str): The name of the column for which to display unique values.

    Returns:
    List: A list of unique values in the specified column.
    """
    print(f"---------- {column} -----------")
    unique_values: List = df[column].unique()
    print(unique_values)
    print(f"Total unique values in column {column}: {len(unique_values)} \n")
    return unique_values


def specificity(y_true, y_pred):
    """
    Calculate specificity as a custom metric.
    """
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    specificity_value = true_negatives / (
        true_negatives + false_positives + K.epsilon()
    )
    return specificity_value
