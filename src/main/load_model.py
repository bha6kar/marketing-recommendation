import pickle
from typing import Callable, Union

import joblib
from tensorflow.keras.models import load_model
from src.main.util import specificity


def load_model_from_file(file_path: str) -> Union[joblib, pickle, load_model]:
    """
    Loads a machine learning model from a file.

    Parameters:
    - file_path (str): The path to the file containing the saved model.

    Returns:
    - Union[joblib, pickle, load_model]: The loaded machine learning model.
    """
    extensions_mapping: dict[str, Callable] = {
        "joblib": joblib.load,
        "pkl": pickle.load,
        "keras": load_model,
    }

    file_extension: str = file_path.split(".")[-1]

    if file_extension not in extensions_mapping:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    if file_extension == "keras":
        return extensions_mapping[file_extension](
            file_path, custom_objects={"specificity": specificity}
        )
    else:
        with open(file_path, "rb") as file:
            return extensions_mapping[file_extension](file)
