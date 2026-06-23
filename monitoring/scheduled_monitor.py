# monitoring/scheduled_monitor.py
"""
Monitoreo de calidad de datos y deriva (drift) con Evidently AI.
Adaptado para el proyecto NVIDIA MLOps.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import os
import json
import logging
import glob

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurar MLflow
mlflow.set_tracking_uri("file:./mlflow_runs")

# ============================================
# FUNCIONES ADAPTADAS PARA TU PROYECTO
# ============================================

def load_data():
    """
    Carga los datos procesados más recientes.
    Busca automáticamente el archivo Parquet más reciente en data/processed/
    """
    try:
        processed_files = glob.glob("data/processed/*.parquet")
        
        if not processed_files:
            logger.error("❌ No hay archivos Parquet en data/processed/")
            return None
        
        latest_file = max(processed_files, key=os.path.getctime)
        logger.info(f"📂 Cargando: {latest_file}")
        
        df = pd.read_parquet(latest_file)
        logger.info(f"✅ Datos cargados: {len(df)} registros")
        logger.info(f"📊 Columnas: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error cargando datos: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_data(df):
    """
    Prepara los datos para monitoreo.
    Separa en referencia (entrenamiento) y actual (producción).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].copy()
    
    if "target" in df_numeric.columns:
        df_numeric.drop(columns=["target"], inplace=True)
    
    exclude_cols = ["Date"]
    for col in exclude_cols:
        if col in df_numeric.columns:
            df_numeric.drop(columns=[col], inplace=True)
    
    split_point = min(300, len(df_numeric) // 2)
    reference = df_numeric.head(split_point)
    current = df_numeric.tail(split_point)
    
    logger.info(f"📊 Datos de referencia: {len(reference)} registros")
    logger.info(f"📊 Datos actuales: {len(current)} registros")
    logger.info(f"📊 Features monitoreadas: {list(reference.columns)}")
    
    return reference, current


def run_monitoring_evidently(reference, current):
    """
    Ejecuta monitoreo con Evidently AI.
    Genera reporte HTML y extrae métricas de drift.
    """
    try:
        logger.info("📊 Generando reporte Evidently...")
        
        report = Report(
            metrics=[
                DataDriftPreset(),
                DataQualityPreset()
            ]
        )
        
        report.run(
            reference_data=reference,
            current_data=current
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("monitoring/reports", exist_ok=True)
        
        report_path = f"monitoring/reports/evidently_report_{timestamp}.html"
        report.save_html(report_path)
        logger.info(f"📄 Reporte guardado: {report_path}")
        
        result = report.as_dict()
        
        drift_metrics = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift": False,
            "drift_share": 0.0,
            "number_of_drifted_columns": 0,
            "total_columns": len(reference.columns)
        }
        
        try:
            metrics = result.get("metrics", [])
            for metric in metrics:
                metric_name = metric.get("metric", "")
                if "DatasetDriftMetric" in metric_name or "DataDrift" in metric_name:
                    metric_result = metric.get("result", {})
                    drift_metrics["dataset_drift"] = metric_result.get("dataset_drift", False)
                    drift_metrics["drift_share"] = metric_result.get("drift_share", 0.0)
                    drift_metrics["number_of_drifted_columns"] = metric_result.get("number_of_drifted_columns", 0)
                    break
        except Exception as e:
            logger.warning(f"⚠️ Error extrayendo métricas: {e}")
        
        metrics_path = f"monitoring/reports/metrics_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(drift_metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Métricas guardadas: {metrics_path}")
        
        return drift_metrics, report_path
        
    except Exception as e:
        logger.error(f"❌ Error en monitoreo: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """
    Función principal del monitoreo.
    """
    logger.info("=" * 50)
    logger.info("🚀 Iniciando monitoreo de modelo NVIDIA")
    logger.info("=" * 50)
    
    try:
        import evidently
        logger.info(f"📦 Evidently version: {evidently.__version__}")
    except Exception:
        pass
    
    df = load_data()
    if df is None:
        logger.error("❌ No se pudieron cargar los datos")
        return
    
    reference, current = prepare_data(df)
    if reference.empty or current.empty:
        logger.error("❌ Datos insuficientes para monitoreo")
        return
    
    drift_metrics, report_path = run_monitoring_evidently(reference, current)
    
    if drift_metrics:
        logger.info("\n" + "=" * 50)
        logger.info("📊 RESUMEN DE MONITOREO")
        logger.info("=" * 50)
        
        logger.info(f"📊 Drift share: {drift_metrics['drift_share']:.2%}")
        logger.info(f"📊 Columnas con drift: {drift_metrics['number_of_drifted_columns']}/{drift_metrics['total_columns']}")
        
        if drift_metrics["dataset_drift"]:
            logger.warning("⚠️ Se detectó drift en el dataset")
        else:
            logger.info("✅ No se detectó drift significativo")
        
        logger.info(f"📄 Reporte: {report_path}")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()