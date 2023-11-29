from typing import List

import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def merge_data(attribution_data_path: str, user_feature_data_path: str) -> DataFrame:
    """
    Merge attribution path data and user feature data based on the 'path_id'
    column.

    Parameters:
    - attribution_data_path (str): File path to the attribution path data in
    feather format.
    - user_feature_data_path (str): File path to the user feature data in
    feather format.

    Returns:
    DataFrame: Merged data containing both attribution path and
    user feature information.
    """
    # Read feather data
    attribution_path_data: DataFrame = pd.read_feather(attribution_data_path)
    user_feature_data: DataFrame = pd.read_feather(user_feature_data_path)

    # Reset index
    attribution_path_data.reset_index(drop=True, inplace=True)
    user_feature_data.reset_index(drop=True, inplace=True)

    # Merge datasets
    merged_data: DataFrame = pd.merge(
        attribution_path_data, user_feature_data, on="path_id", how="inner"
    )
    return merged_data


def preprocess_data(
    df: DataFrame,
    save_path: str = "data/processed/preprocessed_data.feather",
    imputation_strategy: str = "mean",
) -> DataFrame:
    """
    Processes the input DataFrame for a machine learning pipeline.

    Parameters:
    - df (DataFrame): The input DataFrame containing raw data.
    - save_path (str, optional): The file path to save the preprocessed data
    as a Feather file.
    - imputation_strategy (str, optional): The strategy to use for imputing
    missing values.
    Can be one of {'mean', 'median', 'constant', 'most_frequent'}.

    Returns:
    - DataFrame: The preprocessed DataFrame.
    """
    print("---------- Preprocessing data -----------")
    # Drop unnecessary columns
    columns_to_drop: List[str] = [
        "clicked_city",
        "viewed_city",
        "viewed_beach",
    ]
    print(f"Dropping unnecessary features: {columns_to_drop}")
    cleaned_data: DataFrame = df.drop(
        columns_to_drop,
        axis=1,
    )

    # Encode boolean columns
    bool_columns: List[str] = ["saw_offer_summary", "saw_panda"]
    cleaned_data[bool_columns] = cleaned_data[bool_columns].astype(int)

    # Encode object columns to boolean columns
    bool_object_columns: List[str] = [
        "saw_brand",
        "saw_organic",
        "saw_direct",
    ]
    # Define a mapping for encoding
    mapping: dict[bool | None, int] = {True: 1, False: 0, None: -1}

    # Using the map function
    for column in bool_object_columns:
        cleaned_data[column] = cleaned_data[column].map(mapping).fillna(-1)

    # Encode categorical columns
    label_encoder = LabelEncoder()
    categorical_columns: List[str] = [
        "most_common_landing_page",
    ]
    for column in categorical_columns:
        cleaned_data[column] = label_encoder.fit_transform(
            cleaned_data[column].astype(str)
        )

    # Retrieve original column names before imputation
    original_feature_names: List[str] = cleaned_data.columns.tolist()

    # Impute missing values
    imputer = SimpleImputer(strategy=imputation_strategy)
    imputed_data = DataFrame(imputer.fit_transform(cleaned_data))

    # Restore original column names
    imputed_data.columns = original_feature_names

    # Create new features
    imputed_data["time_to_book_week"] = imputed_data["time_delta_in_days"] / 7

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_columns: List[str] = [
        "time_to_book_week",
        "adults",
        "children",
        "nights",
    ]
    imputed_data[numerical_columns] = scaler.fit_transform(
        imputed_data[numerical_columns]
    )

    # Save preprocessed data
    imputed_data.to_feather(save_path)

    return imputed_data
