import pickle

import joblib
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Attention, Dense, GlobalAveragePooling1D, Input
from keras.models import Model, Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
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
    print("Training Random Forest Classifier with GridSearchCV...")

    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # Define the parameter grid for GridSearchCV
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        # Add other parameters to tune
    }

    # Create a RandomForestClassifier
    rf_model = RandomForestClassifier(class_weight=class_weight_dict)

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=rf_model, param_grid=param_grid, scoring="accuracy", cv=3
    )
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Train the model with the best parameters
    best_rf_model = RandomForestClassifier(
        class_weight=class_weight_dict, **best_params
    )
    best_rf_model.fit(X_train, y_train)

    # Save the best model
    save_model(best_rf_model, f"{save_path}/model_rf.joblib")

    return best_rf_model


def get_model_LogisticRegression(
    X_train, y_train, save_path="models"
) -> LogisticRegression:
    print("Training Logistic Regression with GridSearchCV...")

    # Define the parameter grid for GridSearchCV
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"]
        # Add other parameters to tune
    }

    # Create a LogisticRegression model
    lr_model = LogisticRegression(max_iter=1000)

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=lr_model, param_grid=param_grid, scoring="accuracy", cv=3
    )
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Train the model with the best parameters
    best_lr_model = LogisticRegression(max_iter=1000, **best_params)
    best_lr_model.fit(X_train, y_train)

    # Save the best model
    save_model(best_lr_model, f"{save_path}/model_lr.joblib")

    return best_lr_model


def get_model_XGBoost(X_train, y_train, save_path="models") -> XGBClassifier:
    print("Training XGBoost with GridSearchCV...")

    # Define the parameter grid for GridSearchCV
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        # Add other parameters to tune
    }

    # Create an XGBoost classifier
    xgb_model = XGBClassifier()

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=xgb_model, param_grid=param_grid, scoring="accuracy", cv=5
    )
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Train the model with the best parameters
    best_xgb_model = XGBClassifier(**best_params)
    best_xgb_model.fit(X_train, y_train)

    # Save the best model
    save_model(best_xgb_model, f"{save_path}/model_xgboost.pkl")

    return best_xgb_model


def get_model_LSTM(X_train, y_train, save_path="models"):
    # Reshape X_train to a 3D array (samples, time steps, features)
    X_train_3d = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])

    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    def create_model(num_units=50):
        model = Sequential()
        model.add(
            LSTM(num_units, input_shape=(X_train_3d.shape[1], X_train_3d.shape[2]))
        )
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        return model

    # Create KerasClassifier wrapper for use with GridSearchCV
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

    # Define the parameter grid
    param_grid = {
        "num_units": [32, 64, 128, 256],
    }

    # Perform grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="f1", cv=3)
    grid_result = grid.fit(X_train_3d, y_train, class_weight=class_weight_dict)

    # Get the best parameters from the grid search
    best_params = grid_result.best_params_

    # Use the best parameters to create the final model
    final_model = create_model(num_units=best_params["num_units"])

    # Train the final model
    final_model.fit(
        X_train_3d,
        y_train,
        epochs=10,  # You can adjust the number of epochs
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    )

    # Save the final model
    final_model.save(f"{save_path}/model_lstm.keras")

    return final_model


def get_model_with_attention(X_train, y_train, save_path="models") -> Model:
    # Reshape X_train to a 3D array (samples, time steps, features)
    X_train_3d = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])

    # Build the LSTM model with attention
    inputs = Input(shape=(X_train_3d.shape[1], X_train_3d.shape[2]))

    # LSTM layer
    lstm_out = LSTM(50, return_sequences=True)(inputs)

    # Attention layer
    attention = Attention()([lstm_out, lstm_out])

    # Global average pooling to reduce the sequence length
    context = GlobalAveragePooling1D()(attention)

    # Dense layer for classification
    output = Dense(1, activation="sigmoid")(context)

    # Build the model
    model = Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", specificity],
    )

    # Train the model
    model.fit(X_train_3d, y_train, epochs=10, batch_size=32)

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
