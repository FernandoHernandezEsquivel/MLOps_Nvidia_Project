import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_data(file_path):
    """Load and clean CSV data"""
    df = pd.read_csv(file_path)
    
    print(f"📂 Loading: {file_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert Date to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Sort by date
        df = df.sort_values('Date')
    
    # Clean numeric columns
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            # If it's string, clean it
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('[,]', '', regex=True)
                df[col] = df[col].str.replace('[\$,]', '', regex=True)
                df[col] = df[col].str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing data
    df = df.dropna(subset=['Close', 'High', 'Low', 'Open', 'Volume'])
    
    print(f"✅ Loaded {len(df)} records")
    return df

def create_features(df_raw):
    """Create technical features"""
    print("🔧 Creating features...")
    df = df_raw.copy()
    
    # Ensure data is sorted by date
    if 'Date' in df.columns:
        df = df.sort_values('Date')
    
    # Returns
    df['Return_1d'] = df['Close'].pct_change()
    df['Return_5d'] = df['Close'].pct_change(periods=5)
    df['Return_10d'] = df['Close'].pct_change(periods=10)
    
    # Moving averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Volatility
    df['Volatility_10d'] = df['Return_1d'].rolling(window=10).std()
    
    # Volume features
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
    
    # Price ratios
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Drop rows with NaN
    df = df.dropna()
    
    print(f"✅ Features created. Total: {len(df)} records")
    return df

def save_processed_data(df):
    """Save processed data"""
    os.makedirs('data/processed', exist_ok=True)
    
    # Prepare data for saving
    df_save = df.copy()
    if 'Date' in df_save.columns:
        # Already have Date column
        pass
    elif df_save.index.name == 'Date' or isinstance(df_save.index, pd.DatetimeIndex):
        # Reset index to get Date as column
        df_save = df_save.reset_index()
        if 'index' in df_save.columns:
            df_save.rename(columns={'index': 'Date'}, inplace=True)
    
    # Save as Parquet
    parquet_file = 'data/processed/nvda_processed_20260619.parquet'
    df_save.to_parquet(parquet_file, index=False)
    print(f"✅ Data saved to: {parquet_file}")
    
    # Save as CSV for easy inspection
    csv_file = 'data/processed/nvda_processed_20260619.csv'
    df_save.to_csv(csv_file, index=False)
    print(f"✅ CSV saved to: {csv_file}")
    
    # Also save just the features summary
    feature_summary = pd.DataFrame({
        'Feature': df_save.columns,
        'Type': df_save.dtypes.astype(str),
        'Non-Null Count': df_save.count(),
        'Null Count': df_save.isnull().sum()
    })
    feature_summary.to_csv('data/processed/feature_summary.csv', index=False)
    print(f"✅ Feature summary saved to: data/processed/feature_summary.csv")

# Main execution
if __name__ == "__main__":
    try:
        # Load data
        df_raw = load_data('data/raw/nvda_raw_20260619.csv')
        
        # Create features
        df_processed = create_features(df_raw)
        
        # Save processed data
        save_processed_data(df_processed)
        
        # Display summary
        print("\n" + "="*50)
        print("📊 Data Summary")
        print("="*50)
        print(f"Total records: {len(df_processed)}")
        print(f"Total features: {df_processed.shape[1]}")
        if 'Date' in df_processed.columns:
            print(f"Date range: {df_processed['Date'].min()} to {df_processed['Date'].max()}")
        print("\nFirst 5 rows:")
        print(df_processed.head())
        print("\nColumn names:")
        print(df_processed.columns.tolist())
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()