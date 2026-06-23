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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ CONFIGURACIÓN MLFLOW ============
# Método 1: Usar ruta absoluta (más seguro)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DIR = os.path.join(BASE_DIR, "mlflow_runs")

print(f"\n🔍 Configuración MLflow:")
print(f"   📁 Directorio base: {BASE_DIR}")
print(f"   📁 MLflow dir: {MLFLOW_DIR}")
print(f"   📁 Existe: {os.path.exists(MLFLOW_DIR)}")

if not os.path.exists(MLFLOW_DIR):
    print(f"   ❌ Error: Directorio no encontrado")
    print(f"   💡 Asegúrate de ejecutar desde: {BASE_DIR}")
    sys.exit(1)

# Configurar MLflow
mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
logger.info(f"✅ MLflow configurado: {mlflow.get_tracking_uri()}")

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

# ============ CARGA DEL MODELO ============
MODEL_NAME = "NVIDIA_Price_Predictor"
MODEL_STAGE = "Production"

def load_model():
    """Carga el modelo desde MLflow"""
    try:
        # Primero, verificar modelos disponibles
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()
        
        print("\n📦 Modelos disponibles:")
        for m in models:
            print(f"   - {m.name}")
            for v in m.latest_versions:
                print(f"     Versión {v.version}: {v.current_stage}")
        
        # Intentar cargar el modelo
        model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
        print(f"\n🔄 Cargando: {model_uri}")
        
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        print(f"❌ Error cargando de etapa '{MODEL_STAGE}': {e}")
        
        try:
            # Intentar con la última versión
            model_uri = f"models:/{MODEL_NAME}/latest"
            print(f"🔄 Intentando: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            print("✅ Modelo cargado (última versión)")
            return model
            
        except Exception as e2:
            print(f"❌ Error final: {e2}")
            raise RuntimeError(f"No se pudo cargar el modelo '{MODEL_NAME}'")

# ============ FASTAPI APP ============
app = FastAPI(
    title="NVIDIA Stock Price Predictor API",
    description="API para predicción de precios de NVIDIA",
    version="1.0.0"
)

model = None
model_version = None

@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar"""
    global model, model_version
    
    try:
        model = load_model()
        
        # Obtener versión
        try:
            client = mlflow.tracking.MlflowClient()
            latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            model_version = latest[0].version if latest else "latest"
        except:
            model_version = "unknown"
            
        print(f"✅ Modelo {MODEL_NAME} versión {model_version} cargado")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        model = None

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
        # Convertir datos a DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
        # ✅ IMPORTANTE: Reordenar columnas en el orden que espera el modelo
        # Los modelos XGBoost requieren que las columnas estén en el mismo orden
        # que durante el entrenamiento
        expected_columns = model.feature_names_in_
        input_df = input_df[expected_columns]
        
        # Hacer predicción
        prediction = model.predict(input_df)[0]
        
        # Preparar respuesta
        response = ModelPrediction(
            predicted_price=float(prediction),
            model_version=str(model_version) if model_version else "latest",
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