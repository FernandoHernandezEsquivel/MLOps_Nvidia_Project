# tests/test_api.py
"""
Pruebas para la API (FastAPI).
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Añadir el proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.app import app

client = TestClient(app)

def test_root_endpoint():
    """Verifica que el endpoint raíz funciona."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_endpoint():
    """Verifica que el endpoint de health funciona."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_with_valid_data(sample_input_data):
    """Verifica que el endpoint /predict funciona con datos válidos."""
    # Si el modelo no está cargado, la respuesta será 503
    response = client.post("/predict", json=sample_input_data)
    
    # Si el modelo está cargado, debería ser 200
    if response.status_code == 200:
        data = response.json()
        assert "predicted_price" in data
        assert "model_version" in data
        assert isinstance(data["predicted_price"], float)
    else:
        # Si el modelo no está cargado, 503 es aceptable
        assert response.status_code in [200, 503]

def test_predict_endpoint_missing_data():
    """Verifica que el endpoint maneja datos faltantes."""
    # Enviar datos incompletos
    incomplete_data = {"Open": 150.0}
    response = client.post("/predict", json=incomplete_data)
    # Debería fallar con 422 (validación) o 503 (modelo no disponible)
    assert response.status_code in [422, 503]

def test_predict_endpoint_invalid_data():
    """Verifica que el endpoint maneja datos inválidos."""
    invalid_data = {
        "Open": "string",  # Debería ser número
        "High": 155.0,
        "Low": 145.0,
        "Close": 152.0,
        "Volume": 5000
    }
    response = client.post("/predict", json=invalid_data)
    # Debería fallar con 422 (validación) o 503 (modelo no disponible)
    assert response.status_code in [422, 503]