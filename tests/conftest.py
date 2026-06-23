# tests/conftest.py
"""
Configuración para pruebas unitarias.
Contiene fixtures reutilizables.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_data():
    """
    Genera datos de prueba con las mismas columnas que tu modelo.
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.randn(200) * 10 + 150,
        'High': np.random.randn(200) * 10 + 155,
        'Low': np.random.randn(200) * 10 + 145,
        'Close': np.random.randn(200) * 10 + 150,
        'Volume': np.random.randint(1000, 10000, 200),
        'Return_1d': np.random.randn(200) * 0.02,
        'Return_5d': np.random.randn(200) * 0.02,
        'Return_10d': np.random.randn(200) * 0.02,
        'MA_10': np.random.randn(200) * 10 + 150,
        'MA_50': np.random.randn(200) * 10 + 150,
        'Volatility_10d': np.random.randn(200) * 0.01 + 0.02,
        'Volume_MA_10': np.random.randint(1000, 10000, 200),
        'Volume_Ratio': np.random.randn(200) * 0.5 + 1,
        'High_Low_Ratio': np.random.randn(200) * 0.05 + 1.02,
        'Close_Open_Ratio': np.random.randn(200) * 0.05 + 1.01
    })
    
    return df

@pytest.fixture
def sample_input_data():
    """
    Datos de entrada para la API (formato ModelInput).
    """
    return {
        "Open": 150.0,
        "High": 155.0,
        "Low": 145.0,
        "Close": 152.0,
        "Volume": 5000,
        "Return_1d": 0.01,
        "Return_5d": 0.05,
        "Return_10d": 0.08,
        "MA_10": 151.0,
        "MA_50": 148.0,
        "Volatility_10d": 0.02,
        "Volume_MA_10": 4800,
        "Volume_Ratio": 1.04,
        "High_Low_Ratio": 1.07,
        "Close_Open_Ratio": 1.01
    }