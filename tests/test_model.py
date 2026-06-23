# tests/test_model.py
"""
Pruebas para el modelo.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

def test_xgboost_trains_without_errors(sample_data):
    """Verifica que XGBoost entrena sin errores."""
    from src.feature_engineering import create_features
    df = create_features(sample_data)
    
    # Features que usa tu modelo (15 features)
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return_1d', 'Return_5d', 'Return_10d',
        'MA_10', 'MA_50', 'Volatility_10d',
        'Volume_MA_10', 'Volume_Ratio',
        'High_Low_Ratio', 'Close_Open_Ratio'
    ]
    
    X = df[features]
    y = df['Close'].shift(-1)  # Target: precio del día siguiente
    
    # Eliminar NaN
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert isinstance(predictions, np.ndarray)

def test_xgboost_predictions_are_reasonable(sample_data):
    """Verifica que las predicciones son números reales."""
    from src.feature_engineering import create_features
    df = create_features(sample_data)
    
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return_1d', 'Return_5d', 'Return_10d',
        'MA_10', 'MA_50', 'Volatility_10d',
        'Volume_MA_10', 'Volume_Ratio',
        'High_Low_Ratio', 'Close_Open_Ratio'
    ]
    
    X = df[features]
    y = df['Close'].shift(-1)
    
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    assert np.issubdtype(predictions.dtype, np.number)
    assert np.abs(predictions).mean() < 1000