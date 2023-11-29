import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.main.feature_selection import feature_selection, train_val_test_split


@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    data = {
        "feature_1": [1, 2, 3, 4, 5],
        "feature_2": [5, 4, 3, 2, 1],
        "time_delta_in_days": [10, 20, 30, 40, 50],
        "distance_to_last": [1, 2, 3, 4, 5],
        "has_booked": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    return df


def test_feature_selection(sample_data):
    df = sample_data
    selected_feature_df = feature_selection(df)

    expected_columns = ["feature_1", "feature_2", "has_booked"]
    assert_frame_equal(selected_feature_df, df[expected_columns])


def test_train_val_test_split(sample_data):
    df = sample_data
    X_train, X_test, X_val, y_train, y_test, y_val = train_val_test_split(df)

    assert len(X_train) == 3
    assert len(X_test) == 1
    assert len(X_val) == 1
    assert len(y_train) == 3
    assert len(y_test) == 1
    assert len(y_val) == 1
