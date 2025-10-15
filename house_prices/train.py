import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def build_model(data: pd.DataFrame) -> dict[str, float]:
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    num_cols = X_train.select_dtypes(include=np.number).columns
    cat_cols = X_train.select_dtypes(exclude=np.number).columns
    X_train[cat_cols] = X_train[cat_cols].astype(str)
    X_test[cat_cols] = X_test[cat_cols].astype(str)
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_train_processed = np.concatenate([X_train_num, X_train_cat], axis=1)
    X_test_num = scaler.transform(X_test[num_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])
    X_test_processed = np.concatenate([X_test_num, X_test_cat], axis=1)
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    y_pred = np.maximum(model.predict(X_test_processed), 0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    joblib.dump(model, "models/model.joblib")
    joblib.dump(encoder, "models/encoder.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print(f"Model trained and saved successfully. RMSE: {rmse:.4f}")
    return {"rmse": rmse}
