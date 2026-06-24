# frontend/streamlit_app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import time

# ============ CONFIGURACIÓN ============
# 🔥 ACTUALIZA ESTA URL CON LA DE TU API EN RENDER
API_URL = "https://nvidia-price-api.onrender.com"  # <-- CAMBIA ESTO

st.set_page_config(
    page_title="NVIDIA Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NVIDIA Stock Predictor - MLOps Dashboard")
st.markdown("*Dashboard interactivo para predecir el precio de cierre de NVIDIA*")

# ============ SIDEBAR ============
st.sidebar.header("⚙️ Configuración")
api_url = st.sidebar.text_input("URL de la API", value=API_URL)

# Verificar conexión con la API
if st.sidebar.button("🔌 Verificar API"):
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            st.sidebar.success("✅ API conectada correctamente")
            data = response.json()
            st.sidebar.info(f"📦 Modelo: {data.get('model_name', 'N/A')} versión {data.get('model_version', 'N/A')}")
        else:
            st.sidebar.error(f"❌ Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("❌ No se pudo conectar a la API. Verifica la URL.")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

# ============ DATOS HISTÓRICOS ============
@st.cache_data(ttl=3600)  # ✅ CAMBIADO: st.cache_data en lugar de st.cache
def load_historical_data():
    """Carga datos históricos de NVIDIA."""
    try:
        df = yf.download("NVDA", period="6mo", interval="1d")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return pd.DataFrame()

df_hist = load_historical_data()
if df_hist.empty:
    st.warning("No se pudieron cargar datos históricos. Verifica tu conexión a internet.")
    st.stop()

# ============ GRÁFICO Y PREDICCIÓN ============
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Precio Histórico (Últimos 6 meses)")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_hist['Date'],
        y=df_hist['Close'],
        mode='lines',
        name='Precio de Cierre',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_hist['Date'],
        y=df_hist['High'],
        mode='lines',
        name='Máximo',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df_hist['Date'],
        y=df_hist['Low'],
        mode='lines',
        name='Mínimo',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)  # ✅ Se mantiene por ahora

with col2:
    st.subheader("🔮 Hacer una Predicción")
    st.markdown("Ingresa los datos del día actual:")
    
    last_open = float(df_hist['Open'].iloc[-1].item())
    last_high = float(df_hist['High'].iloc[-1].item())
    last_low = float(df_hist['Low'].iloc[-1].item())
    last_close = float(df_hist['Close'].iloc[-1].item())
    last_volume = int(df_hist['Volume'].iloc[-1].item())
    
    open_price = st.number_input("Open", value=last_open, format="%.2f")
    high_price = st.number_input("High", value=last_high, format="%.2f")
    low_price = st.number_input("Low", value=last_low, format="%.2f")
    close_price = st.number_input("Close", value=last_close, format="%.2f")
    volume = st.number_input("Volume", value=last_volume)
    
    if st.button("🚀 Predecir Precio de Cierre"):
        # Calcular features
        if len(df_hist) >= 2:
            return_1d = (df_hist['Close'].iloc[-1].item() / df_hist['Close'].iloc[-2].item() - 1)
        else:
            return_1d = 0.0
        
        if len(df_hist) >= 6:
            return_5d = (df_hist['Close'].iloc[-1].item() / df_hist['Close'].iloc[-6].item() - 1)
        else:
            return_5d = 0.0
        
        if len(df_hist) >= 11:
            return_10d = (df_hist['Close'].iloc[-1].item() / df_hist['Close'].iloc[-11].item() - 1)
        else:
            return_10d = 0.0
        
        ma_10 = float(df_hist['Close'].rolling(10).mean().iloc[-1].item()) if len(df_hist) >= 10 else close_price
        ma_50 = float(df_hist['Close'].rolling(50).mean().iloc[-1].item()) if len(df_hist) >= 50 else close_price
        
        returns = df_hist['Close'].pct_change()
        volatility_10d = float(returns.rolling(10).std().iloc[-1].item()) if len(df_hist) >= 10 else 0.02
        
        volume_ma_10 = float(df_hist['Volume'].rolling(10).mean().iloc[-1].item()) if len(df_hist) >= 10 else volume
        volume_ratio = volume / volume_ma_10 if volume_ma_10 > 0 else 1.0
        high_low_ratio = high_price / low_price if low_price > 0 else 1.0
        close_open_ratio = close_price / open_price if open_price > 0 else 1.0
        
        payload = {
            "Open": float(open_price),
            "High": float(high_price),
            "Low": float(low_price),
            "Close": float(close_price),
            "Volume": int(volume),
            "Return_1d": float(return_1d),
            "Return_5d": float(return_5d),
            "Return_10d": float(return_10d),
            "MA_10": float(ma_10),
            "MA_50": float(ma_50),
            "Volatility_10d": float(volatility_10d),
            "Volume_MA_10": float(volume_ma_10),
            "Volume_Ratio": float(volume_ratio),
            "High_Low_Ratio": float(high_low_ratio),
            "Close_Open_Ratio": float(close_open_ratio)
        }
        
        try:
            with st.spinner("⏳ Prediciendo..."):
                response = requests.post(f"{api_url}/predict", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["predicted_price"]
                model_version = result.get("model_version", "latest")
                
                st.success(f"💲 Precio de cierre predicho: **${prediction:.2f}**")
                st.info(f"📌 Versión del modelo: {model_version}")
                
                actual_close = float(df_hist['Close'].iloc[-1].item())
                difference = prediction - actual_close
                diff_pct = (difference / actual_close) * 100
                
                st.write("---")
                st.write(f"**📊 Precio Actual:** ${actual_close:.2f} | **🔮 Predicción:** ${prediction:.2f} | **📈 Diferencia:** {diff_pct:.2f}%")
            else:
                st.error(f"❌ Error en la API: {response.status_code}")
                st.json(response.json() if response.content else {})
        except requests.exceptions.ConnectionError:
            st.error("❌ No se pudo conectar a la API. Verifica la URL.")
            st.info(f"📡 URL configurada: {api_url}")
        except requests.exceptions.Timeout:
            st.error("❌ Tiempo de espera agotado. La API tardó demasiado en responder.")
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")

# ============ ESTADO DEL SISTEMA ============
st.markdown("---")
st.subheader("📊 Estado del Sistema")

col3, col4, col5, col6 = st.columns(4)

with col3:
    current_price = float(df_hist['Close'].iloc[-1].item())
    previous_price = float(df_hist['Close'].iloc[-2].item()) if len(df_hist) >= 2 else current_price
    price_change = ((current_price / previous_price) - 1) * 100 if previous_price > 0 else 0
    st.metric("💰 Precio Actual", f"${current_price:.2f}", delta=f"{price_change:.2f}%")

with col4:
    volume_today = int(df_hist['Volume'].iloc[-1].item())
    volume_avg = float(df_hist['Volume'].mean().item())
    volume_ratio = volume_today / volume_avg if volume_avg > 0 else 1
    st.metric("📊 Volumen", f"{volume_today:,.0f}", delta=f"{volume_ratio:.2f}x promedio")

with col5:
    if len(df_hist) >= 15:
        delta = df_hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        rsi_value = float(rsi.iloc[0].item()) if hasattr(rsi.iloc[0], 'item') else 50
        st.metric("📈 RSI (14 días)", f"{rsi_value:.1f}")
    else:
        st.metric("📈 RSI (14 días)", "N/A")

with col6:
    st.metric("📅 Última actualización", datetime.now().strftime("%H:%M:%S"))

# ============ TABLA DE DATOS ============
st.markdown("---")
st.subheader("📋 Datos Recientes")
with st.expander("Ver tabla de datos históricos"):
    styled_df = df_hist.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].style.format({
        'Date': lambda x: x.strftime('%Y-%m-%d'),
        'Open': '${:.2f}',
        'High': '${:.2f}',
        'Low': '${:.2f}',
        'Close': '${:.2f}',
        'Volume': '{:,.0f}'
    })
    st.dataframe(styled_df)