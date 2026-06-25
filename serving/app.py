# serving/app.py
import os
import sys
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging
import glob

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ CONFIGURACIÓN MLFLOW ============
# Ruta absoluta del proyecto
BASE_DIR = os.getenv("APP_DIR", "/app")
MLFLOW_DIR = os.path.join(BASE_DIR, "mlflow_runs")

# Crear el directorio si no existe
os.makedirs(MLFLOW_DIR, exist_ok=True)

print(f"\n🔍 Configuración MLflow:")
print(f"   📁 Directorio base: {BASE_DIR}")
print(f"   📁 MLflow dir: {MLFLOW_DIR}")
print(f"   📁 Existe: {os.path.exists(MLFLOW_DIR)}")

# Configurar MLflow
mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
logger.info(f"✅ MLflow configurado: {mlflow.get_tracking_uri()}")

# ============ NOMBRE DEL MODELO Y ALIAS ============
MODEL_NAME = "NVIDIA_Price_Predictor"
MODEL_ALIAS = "champion"  # Usamos alias en lugar de stage

# ============ CARGA DEL MODELO (UNIFICADA) ============
def load_model():
    try:
        direct_model_path = os.path.join(
            MLFLOW_DIR,
            "504246946478130849",
            "models",
            "m-56df30315e2f4c298c656aa55c3741fd",
            "artifacts"
        )

        print(f"🔄 Cargando modelo desde: {direct_model_path}")

        model = mlflow.sklearn.load_model(direct_model_path)

        print("✅ Modelo cargado correctamente")

        return model, "v4"

    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

        import traceback
        traceback.print_exc()

        return None, None

# ============ INICIALIZACIÓN ============
print("🚀 Iniciando API de predicción NVIDIA...")

# Cargar el modelo al inicio
model, model_version = load_model()

if model is not None:
    print(f"✅ Modelo listo para usar (versión: {model_version})")
else:
    print("❌ NO se pudo cargar el modelo. La API funcionará en modo limitado.")

# ============ ESQUEMA DE DATOS ============
class ModelInput(BaseModel):
    Open: float = Field(..., description="Precio de apertura")
    High: float = Field(..., description="Precio máximo del día")
    Low: float = Field(..., description="Precio mínimo del día")
    Close: float = Field(..., description="Precio de cierre del día")
    Volume: float = Field(..., description="Volumen de operaciones")
    Return_1d: float = Field(..., description="Retorno del día anterior")
    Return_5d: float = Field(..., description="Retorno de hace 5 días")
    Return_10d: float = Field(..., description="Retorno de hace 10 días")
    MA_10: float = Field(..., description="Media móvil de 10 días")
    MA_50: float = Field(..., description="Media móvil de 50 días")
    Volatility_10d: float = Field(..., description="Volatilidad de 10 días")
    Volume_MA_10: float = Field(..., description="Volumen vs media móvil de 10 días")
    Volume_Ratio: float = Field(..., description="Ratio de volumen actual vs promedio")
    High_Low_Ratio: float = Field(..., description="Ratio High/Low del día")
    Close_Open_Ratio: float = Field(..., description="Ratio Close/Open del día")

class ModelPrediction(BaseModel):
    predicted_price: float = Field(..., description="Precio predicho para el próximo día")
    model_version: str = Field(..., description="Versión del modelo utilizado")
    confidence_interval_lower: Optional[float] = Field(None, description="Límite inferior del intervalo de confianza")
    confidence_interval_upper: Optional[float] = Field(None, description="Límite superior del intervalo de confianza")

# ============ FASTAPI APP ============
app = FastAPI(
    title="NVIDIA Stock Price Predictor API",
    description="API para predicción de precios de NVIDIA",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "NVIDIA Stock Price Predictor API",
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "status": "ready" if model is not None else "not_ready"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_name": MODEL_NAME,
        "model_version": model_version
    }

@app.post("/predict", response_model=ModelPrediction)
async def predict(data: ModelInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # ✅ Compatible con Pydantic v1 y v2
        try:
            # Pydantic v2
            input_data = data.model_dump()
        except AttributeError:
            # Pydantic v1
            input_data = data.dict()
            
            # Convertir datos a DataFrame
            input_df = pd.DataFrame([input_data])
        
        # Reordenar columnas en el orden que espera el modelo
        expected_columns = model.feature_names_in_
        input_df = input_df[expected_columns]
        
        # Hacer predicción
        prediction = model.predict(input_df)[0]
        
        # Preparar respuesta
        response = ModelPrediction(
            predicted_price=float(prediction),
            model_version=str(model_version) if model_version else "unknown",
            confidence_interval_lower=float(prediction * 0.95),
            confidence_interval_upper=float(prediction * 1.05)
        )
        
        logger.info(f"📊 Predicción: {response.predicted_price:.4f}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")