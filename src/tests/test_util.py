import numpy as np
import pandas as pd
import pytest

from src.main.util import get_unique_values, specificity


@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    data = {
        "feature_1": [1, 2, 3, 4, 5],
        "feature_2": [5, 4, 3, 2, 1],
        "has_booked": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    return df


def test_get_unique_values(sample_data):
    df = sample_data

    unique_values = get_unique_values(df, "feature_1")

    assert len(unique_values) == 5
    assert 1 in unique_values
    assert 2 in unique_values


def test_specificity():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0.2, 0.8, 0.3, 0.7, 0.4])

    result = specificity(y_true, y_pred)

    expected_result = 0.9999999666666678

    assert result == expected_result
