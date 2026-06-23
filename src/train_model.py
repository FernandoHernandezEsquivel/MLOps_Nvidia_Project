import os
import sys
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ============ MLflow Configuration ============
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlflow_runs"

mlflow.set_tracking_uri("file:./mlflow_runs")

os.makedirs("./mlflow_runs", exist_ok=True)

# ============ Load Data ============
def load_processed_data(file_path):
    """Load processed data"""
    print(f"📂 Cargando datos: {file_path}")
    df = pd.read_parquet(file_path)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
    print(f"✅ Cargados {len(df)} registros")
    print(f"📊 Columnas: {df.columns.tolist()}")
    return df

def prepare_data(df):
    """Prepare features and target for modeling"""
    df = df.copy()
    df['target'] = df['Close'].shift(-1)
    df = df.dropna(subset=['target'])
    
    exclude_cols = ['target', 'Date', 'Return_1d']
    
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols and col != 'Close']
    
    if not feature_cols:
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != 'target']
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"🔧 Features usadas: {feature_cols}")
    print(f"📊 X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, feature_cols

def train_xgboost_model(df):
    """Train XGBoost model with MLflow tracking"""
    
    X, y, feature_cols = prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Datos de entrenamiento: {len(X_train)} registros")
    print(f"📊 Datos de prueba: {len(X_test)} registros")
    
    # IMPORTANT: Set experiment before starting run
    mlflow.set_experiment("NVIDIA_Price_Predictor")
    
    run_name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "model_type": "XGBoost",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "test_size": 0.2,
            "random_state": 42
        }
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        print(f"📊 R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance_path = "feature_importance.csv"
        feature_importance.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(feature_importance_path)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgboost_model",
            registered_model_name="NVIDIA_Price_Predictor",
            input_example=X_train.iloc[:5]
        )
        
        print(f"✅ Modelo guardado en MLflow")
        print(f"🏃 Run ID: {run.info.run_id}")
        
        os.remove(feature_importance_path)
        
        return model, run.info.run_id

def main():
    try:
        df = load_processed_data('data/processed/nvda_processed_20260619.parquet')
        model, run_id = train_xgboost_model(df)
        
        print(f"\n✅ Entrenamiento completado!")
        print(f"📌 Run ID: {run_id}")
        print(f"📊 Para ver los resultados: mlflow ui --backend-store-uri ./mlflow_runs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()