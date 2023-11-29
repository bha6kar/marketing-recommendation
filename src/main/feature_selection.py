from typing import List, Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split


def feature_selection(df: DataFrame) -> DataFrame:
    """
    Performs feature selection on the input DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame containing features.

    Returns:
    - selected_feature_df (DataFrame): The DataFrame with selected features.
    """
    print("---------- Feature selection -----------")
    columns_to_drop: List[str] = [
        "time_delta_in_days",
        "distance_to_last",
    ]
    print(f"Dropping unnecessary features: {columns_to_drop}")
    selected_feature_df: DataFrame = df.drop(columns=columns_to_drop, axis=1)
    return selected_feature_df


def train_val_test_split(df, val_size=0.2, test_size=0.2, random_seed=42) -> Tuple:
    """
    Split the DataFrame into training, validation, and test sets for machine learning.

    Parameters:
    - df (DataFrame): The DataFrame containing the features and the target variable.
    - val_size (float): The proportion of the data to include in the validation split
    (default is 0.2).
    - test_size (float): The proportion of the data to include in the test split
    (default is 0.2).
    - random_seed (int): Seed for reproducibility (default is 42).

    Returns:
    Tuple: A tuple containing X_train, X_test, X_val, y_train, y_test, y_val.
    """
    X = df.drop("has_booked", axis=1)
    y = df["has_booked"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_seed
    )

    return X_train, X_test, X_val, y_train, y_test, y_val
