import pandas as pd
import pytest

from src.main.preprocess_data import merge_data, preprocess_data


@pytest.fixture
def merged_data():
    # Read test data files
    attribution_data_path = "src/tests/data/attribution_path_data.feather"
    user_feature_data_path = "src/tests/data/user_feature_data.feather"
    # Merge data
    return merge_data(attribution_data_path, user_feature_data_path)


def test_merge_data(merged_data):
    assert isinstance(merged_data, pd.DataFrame)
    assert merged_data.shape == (5, 19)


def test_preprocess_data(merged_data):
    preprocessed_data = preprocess_data(merged_data)
    assert isinstance(merged_data, pd.DataFrame)
    assert preprocessed_data.shape == (5, 16)


def test_preprocess_data_with_imputation(merged_data):
    preprocessed_data = preprocess_data(merged_data, imputation_strategy="median")
    assert isinstance(merged_data, pd.DataFrame)
    assert preprocessed_data.shape == (5, 16)
