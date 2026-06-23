# src/data_ingestion.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_nvidia_data(period="2y"):
    """
    Descarga datos históricos de NVIDIA.
    
    Args:
        period (str): Período de tiempo ('1y', '2y', '5y', etc.)
    
    Returns:
        pd.DataFrame: Datos descargados
    """
    print("📥 Descargando datos de NVIDIA...")
    ticker = "NVDA"
    
    # Descargar datos
    df = yf.download(ticker, period=period)
    df.reset_index(inplace=True)  # La fecha pasa a ser una columna normal
    
    print(f"✅ Datos descargados: {len(df)} registros")
    return df

def save_raw_data(df):
    """
    Guarda los datos crudos en la carpeta data/raw/
    """
    # Crear la carpeta si no existe
    os.makedirs("data/raw", exist_ok=True)
    
    # Generar nombre de archivo con la fecha actual
    today = datetime.now().strftime("%Y%m%d")
    filename = f"data/raw/nvda_raw_{today}.csv"
    
    # Guardar como CSV
    df.to_csv(filename, index=False)
    print(f"💾 Datos guardados en: {filename}")
    return filename

if __name__ == "__main__":
    # Si ejecutas este archivo directamente, hace esto:
    df = download_nvidia_data("2y")   # Descarga 2 años de datos
    save_raw_data(df)                 # Los guarda