# pipelines/data_pipeline.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from prefect import flow, task, get_run_logger
from src.data_ingestion import download_nvidia_data, save_raw_data
from src.feature_engineering import create_features, save_processed_data
import sys
import os

@task
def task_download_data():
    """Tarea de Prefect: Descargar datos"""
    logger = get_run_logger()
    logger.info("Iniciando descarga de datos...")
    df = download_nvidia_data("2y")
    logger.info(f"Descargados {len(df)} registros")
    return df

@task
def task_save_raw(df):
    """Tarea de Prefect: Guardar datos crudos"""
    logger = get_run_logger()
    filename = save_raw_data(df)
    logger.info(f"Archivo guardado: {filename}")
    return filename

@task
def task_process_data(raw_file_path):
    """Tarea de Prefect: Procesar datos"""
    logger = get_run_logger()
    logger.info("Cargando último archivo: {raw_file_path}")
    from src.feature_engineering import load_data
    df_raw = load_data(raw_file_path)
    
    logger.info("Creando features...")
    df_processed = create_features(df_raw)
    
    filename = save_processed_data(df_processed)
    logger.info(f"Datos procesados guardados: {filename}")
    return filename

@flow(name="Data Pipeline - NVIDIA")
def data_pipeline():
    """
    Flujo completo de Prefect:
    1. Descarga datos
    2. Guarda crudos
    3. Procesa y guarda procesados
    """
    logger = get_run_logger()
    logger.info("🚀 Iniciando Data Pipeline...")
    
    # Ejecutar tareas en orden
    raw_data = task_download_data()
    raw_file_path = task_save_raw(raw_data)
    processed_data = task_process_data(raw_file_path)
    
    logger.info("✅ Data Pipeline completado con éxito!")

if __name__ == "__main__":
    # Ejecutar el flujo
    data_pipeline()