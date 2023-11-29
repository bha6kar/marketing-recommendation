import numpy as np
from pandas import DataFrame

from src.main.feature_selection import feature_selection
from src.main.preprocess_data import preprocess_data


def predict(model, X: DataFrame) -> np.ndarray:
    """
    Generates predictions using the given machine learning model.

    Parameters:
    - model: The trained machine learning model.
    - X (DataFrame): The input features for prediction.

    Returns:
    - np.ndarray: Predicted labels or probabilities.
    """
    model_type = model.__class__.__name__

    # Preprocess the input data
    X_preprocessed = preprocess_data(X)

    # Perform feature selection
    X_selected = feature_selection(X_preprocessed)
    # Make predictions
    if model_type == "LSTM":
        # Reshape the input for LSTM
        X_reshaped = X_selected.values.reshape(
            X_selected.shape[0], 1, X_selected.shape[1]
        )
        predictions = model.predict(X_reshaped)
    else:
        # For other models, use the predict method directly
        predictions = model.predict(X_selected)

    return predictions
