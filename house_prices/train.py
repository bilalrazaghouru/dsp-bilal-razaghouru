import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import joblib
from house_prices.preprocess import fit_preprocessors, transform_features


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Logarithmic Error."""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def build_model(data: pd.DataFrame) -> dict[str, float]:
    """Train and evaluate Random Forest model, then persist model and transformers."""
    # Split target and features
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    # Separate column types
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    # Split before preprocessing to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit preprocessors (encoder, scaler)
    encoder, scaler = fit_preprocessors(X_train, cat_cols, num_cols)

    # Transform train and test data
    X_train_processed = transform_features(X_train, encoder, scaler, cat_cols, num_cols)
    X_test_processed = transform_features(X_test, encoder, scaler, cat_cols, num_cols)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_processed, y_train)

    # Save model
    joblib.dump(model, "models/model.joblib")

    # Evaluate model
    preds = np.maximum(0, model.predict(X_test_processed))
    rmsle = compute_rmsle(y_test, preds)

    return {"rmsle": round(rmsle, 4)}
