import pandas as pd
import numpy as np
import joblib

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    model = joblib.load("models/model.joblib")
    encoder = joblib.load("models/encoder.joblib")
    scaler = joblib.load("models/scaler.joblib")

    input_data = input_data.fillna(0)
    num_cols = input_data.select_dtypes(include=np.number).columns
    cat_cols = input_data.select_dtypes(exclude=np.number).columns
    input_data[cat_cols] = input_data[cat_cols].astype(str)

    input_num = scaler.transform(input_data[num_cols])
    input_cat = encoder.transform(input_data[cat_cols])
    input_processed = np.concatenate([input_num, input_cat], axis=1)

    predictions = np.maximum(model.predict(input_processed), 0)
    return predictions
