import pandas as pd
import numpy as np
import joblib


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on input data using the saved model and preprocessing objects.

    Steps:
    - Load model, encoder, and scaler
    - Preprocess numeric and categorical columns
    - Generate predictions and return as numpy array

    Args:
        input_data (pd.DataFrame): New dataset to predict on

    Returns:
        np.ndarray: Predicted house prices
    """
    # Load artifacts
    model = joblib.load("models/model.joblib")
    encoder = joblib.load("models/encoder.joblib")
    scaler = joblib.load("models/scaler.joblib")

    # Preprocess input data
    input_data = input_data.fillna(0)
    num_cols = input_data.select_dtypes(include=np.number).columns
    cat_cols = input_data.select_dtypes(exclude=np.number).columns
    input_data[cat_cols] = input_data[cat_cols].astype(str)

    # Apply transformations
    input_num = scaler.transform(input_data[num_cols])
    input_cat = encoder.transform(input_data[cat_cols])
    input_processed = np.concatenate([input_num, input_cat], axis=1)

    # Predict
    predictions = np.maximum(model.predict(input_processed), 0)
    return predictions
