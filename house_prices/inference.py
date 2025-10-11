import pandas as pd
import numpy as np
import joblib
from house_prices.preprocess import transform_features


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """Load saved model and preprocessors to predict on new data."""
    encoder = joblib.load("models/encoder.joblib")
    scaler = joblib.load("models/scaler.joblib")
    model = joblib.load("models/model.joblib")

    cat_cols = input_data.select_dtypes(include="object").columns.tolist()
    num_cols = input_data.select_dtypes(exclude="object").columns.tolist()

    X_processed = transform_features(input_data, encoder, scaler, cat_cols, num_cols)
    preds = np.maximum(0, model.predict(X_processed))
    return preds
