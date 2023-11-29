import pickle

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from xgboost import XGBClassifier

from src.main.util import specificity


def save_model(model, file_path) -> None:
    if isinstance(model, Sequential):  # For Keras models
        model.save(file_path)
    elif "joblib" in str(type(model)):
        joblib.dump(model, file_path)
    else:
        with open(file_path, "wb") as model_file:
            pickle.dump(model, model_file)


def get_model_RandomForestClassifier(
    X_train, y_train, save_path="models"
) -> RandomForestClassifier:
    print("Training Random Forest Classifier...")
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    model = RandomForestClassifier(class_weight=class_weight_dict)
    model.fit(X_train, y_train)
    save_model(model, f"{save_path}/model_rf.joblib")
    return model


def get_model_LogisticRegression(
    X_train, y_train, save_path="models"
) -> LogisticRegression:
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    save_model(model, f"{save_path}/model_lr.joblib")
    return model


def get_model_XGBoost(X_train, y_train, save_path="models") -> XGBClassifier:
    print("Training XGBoost...")

    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    scale_pos_weight = (
        class_weights[0] / class_weights[1]
    )  # ratio of negative to positive samples

    # Initialize XGBoost with the scale_pos_weight parameter
    model = XGBClassifier(scale_pos_weight=scale_pos_weight)

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    save_model(model, f"{save_path}/model_xgboost.joblib")

    return model


def get_model_LSTM(X_train, y_train, save_path="models") -> Sequential:
    print("Training LSTM...")
    # Reshape X_train to a 3D array (samples, time steps, features)
    X_train_3d = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])

    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_3d.shape[1], X_train_3d.shape[2])))
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", specificity],
    )

    # Train the model
    model.fit(
        X_train_3d, y_train, epochs=10, batch_size=32, class_weight=class_weight_dict
    )

    # Save the model
    save_model(model, f"{save_path}/model_lstm.keras")

    return model


def train_models(X_train, y_train) -> None:
    """
    Trains multiple models for binary classification.

    Parameters:
    - X_train: The training features.
    - y_train: The training target variable.

    """
    get_model_LogisticRegression(X_train, y_train)
    get_model_RandomForestClassifier(X_train, y_train)
    get_model_XGBoost(X_train, y_train)
    get_model_LSTM(X_train, y_train)
