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

# Crear el directorio si no existe
os.makedirs(MLFLOW_DIR, exist_ok=True)

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

# ============ VERIFICACIÓN Y CARGA DIRECTA ============
# Función para cargar el modelo directamente desde la carpeta
def load_model_direct():
    """Carga el modelo directamente desde la carpeta mlflow_runs"""
    try:
        # Buscar la versión más reciente
        import glob
        model_paths = glob.glob(f"{MLFLOW_DIR}/models/NVIDIA_Price_Predictor/version-*")
        if model_paths:
            latest_version = sorted(model_paths)[-1]
            print(f"🔄 Cargando modelo desde: {latest_version}")
            model = mlflow.sklearn.load_model(latest_version)
            print(f"✅ Modelo cargado exitosamente desde {latest_version}")
            return model, "direct"
    except Exception as e:
        print(f"❌ Error en carga directa: {e}")
    return None, None

# Intentar cargar el modelo directamente
model, model_version = load_model_direct()
if model is not None:
    print(f"✅ Modelo listo para usar (versión: {model_version})")
else:
    print("❌ No se pudo cargar el modelo")

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
    """Carga el modelo desde MLflow con manejo de errores mejorado."""
    try:
        # Configurar MLflow
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MLFLOW_DIR = os.path.join(BASE_DIR, "mlflow_runs")
        
        print(f"🔍 Configuración MLflow:")
        print(f"   📁 Directorio base: {BASE_DIR}")
        print(f"   📁 MLflow dir: {MLFLOW_DIR}")
        print(f"   📁 Existe: {os.path.exists(MLFLOW_DIR)}")
        
        # Verificar contenido del directorio
        if os.path.exists(MLFLOW_DIR):
            print(f"📂 Contenido de {MLFLOW_DIR}:")
            for item in os.listdir(MLFLOW_DIR):
                print(f"   - {item}")
        
        # Configurar tracking URI
        mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
        
        # Listar modelos disponibles
        client = mlflow.tracking.MlflowClient()
        try:
            models = client.search_registered_models()
            print(f"📦 Modelos encontrados: {len(models)}")
            for model in models:
                print(f"   - {model.name}")
                for version in model.latest_versions:
                    print(f"     Versión {version.version}: {version.stage}")
        except Exception as e:
            print(f"⚠️ No se pudieron listar modelos: {e}")
        
        # Intentar cargar el modelo de producción
        MODEL_NAME = "NVIDIA_Price_Predictor"
        MODEL_STAGE = "Production"
        
        try:
            model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
            print(f"🔄 Intentando cargar: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Modelo cargado desde {MODEL_STAGE}")
            return model, "Production"
        except Exception as e:
            print(f"⚠️ No se pudo cargar de {MODEL_STAGE}: {e}")
            
            # Intentar con la última versión
            try:
                model_uri = f"models:/{MODEL_NAME}/latest"
                print(f"🔄 Intentando con última versión: {model_uri}")
                model = mlflow.sklearn.load_model(model_uri)
                print(f"✅ Modelo cargado (última versión)")
                return model, "latest"
            except Exception as e2:
                print(f"❌ Error final: {e2}")
                return None, None
                
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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
        # Intentar carga directa primero
        model, model_version = load_model_direct()
        if model is not None:
            print(f"✅ Modelo cargado exitosamente (versión: {model_version})")
            return
        
        # Si falla, intentar con MLflow normal
        print("🔄 Intentando carga con MLflow...")
        model, model_version = load_model()
        if model is not None:
            print(f"✅ Modelo cargado exitosamente (versión: {model_version})")
        else:
            print("❌ No se pudo cargar el modelo")
    except Exception as e:
        print(f"❌ Error en startup: {e}")
        model = None
        model_version = None

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
