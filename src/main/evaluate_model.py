from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential


def get_scores(model, X_test, y_test) -> Tuple[float, float]:
    """
    Evaluate the performance of a machine learning model on a test set and
    print specificity, accuracy and F1 score.

    Parameters:
    - model: The trained machine learning model to be evaluated.
    - X_test: The features of the test set.
    - y_test: The true labels of the test set.

    Returns:
    Tuple[float, float]: A tuple containing the specificity, accuracy and F1 score.
    """
    print(f"\n -------{model.__class__.__name__}-------")
    if isinstance(model, Sequential):  # Check if the model is an LSTM model
        print("Reshape for LSTM")
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        y_pred = (model.predict(X_test) >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
    print(f"Unique Predictions: {np.unique(y_pred)}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Specificity: {specificity}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    return specificity, accuracy, f1


def model_selection(models, X_test, y_test, metric):
    """
    Selects the best model based on specificity scores.

    Parameters:
    - models (List[ClassifierMixin]): List of trained classifier models.
    - X_test: The features of the test set.
    - y_test: The true labels of the test set.

    Returns:
    - ClassifierMixin: The best model based on specificity scores.
    """
    all_model_scores: dict = {}
    for model in models:
        specificity, accuracy, f1 = get_scores(model, X_test, y_test)
        match metric:
            case "specificity":
                all_model_scores[model] = specificity
            case "accuracy":
                all_model_scores[model] = accuracy
            case "f1":
                all_model_scores[model] = f1
    print(all_model_scores)
    best_model = max(all_model_scores, key=all_model_scores.get)
    return best_model
