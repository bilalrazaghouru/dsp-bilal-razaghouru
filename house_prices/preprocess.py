import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from typing import Tuple


def fit_preprocessors(
    X_train: pd.DataFrame,
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> Tuple[OneHotEncoder, StandardScaler]:
    """Fit encoder and scaler on training data and save them locally."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    encoder.fit(X_train[categorical_cols])
    scaler.fit(X_train[numerical_cols])

    joblib.dump(encoder, "models/encoder.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    return encoder, scaler


def transform_features(
    X: pd.DataFrame,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> pd.DataFrame:
    """Transform dataset using fitted encoder and scaler."""
    X_cat = encoder.transform(X[categorical_cols])
    X_num = scaler.transform(X[numerical_cols])

    cat_df = pd.DataFrame(
        X_cat, columns=encoder.get_feature_names_out(categorical_cols), index=X.index
    )
    num_df = pd.DataFrame(X_num, columns=numerical_cols, index=X.index)

    X_processed = pd.concat([num_df, cat_df], axis=1)
    return X_processed
